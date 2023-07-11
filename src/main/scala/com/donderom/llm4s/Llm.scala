package com.donderom.llm4s

import java.nio.file.Path

import scala.collection.mutable.ArrayDeque
import scala.util.Try

import fr.hammons.slinc.runtime.given
import fr.hammons.slinc.types.SizeT
import fr.hammons.slinc.{FSet, Ptr, Scope}

private class SlincLlm private[llm4s] (val ctx: Ptr[Any]):
  val llama = FSet.instance[Llama]

  def generate(prompt: String, params: LlmParams): LazyList[String] =
    val lastTokens = new ArrayDeque[Int](ctxSize)

    def tokens(remaining: Int, evaluated: Int): LazyList[String] =
      if remaining != 0 then
        val newPast = if evaluated + 1 > ctxSize then
          val start = ctxSize - ((evaluated - params.keepTokens) / 2) - 1
          evaluate(
            ids = lastTokens.slice(start, lastTokens.size).toArray,
            past = math.max(1, params.keepTokens),
            params = params.context
          )
        else
          val start =
            if lastTokens.size == ctxSize then ctxSize - 1 else evaluated
          evaluate(
            ids = lastTokens.slice(start, lastTokens.size).toArray,
            past = evaluated,
            params = params.context
          )

        val repeatLastTokens =
          if params.sampling.repeatLastTokens < 0 then ctxSize
          else params.sampling.repeatLastTokens
        val lastRepeat = math.min(lastTokens.size, repeatLastTokens)
        val repeatTokens = lastTokens.takeRight(lastRepeat).toArray
        val id = sample(repeatTokens, params.sampling)

        if lastTokens.size == ctxSize then lastTokens.remove(0)
        lastTokens.append(id)

        if lastTokens.lastOption.fold(true)(_ != eosToken) then
          decode(id) #:: tokens(remaining = remaining - 1, evaluated = newPast)
        else close(params.suffix)
      else close(params.suffix)

    val ids = encode(" " + prompt)
    val promptTokens = LazyList.from(ids).tapEach(lastTokens.append).map(decode)
    promptTokens #::: tokens(
      remaining = params.predictTokens,
      evaluated = 0
    )

  lazy val ctxSize: Int = llama.llama_n_ctx(ctx)
  lazy val eosToken: Int = llama.llama_token_eos()
  lazy val vocabSize: Int = llama.llama_n_vocab(ctx)

  val addBos: Int = 1

  def encode(prompt: String): Array[Int] =
    val res = new Array[Int](prompt.size + addBos)
    Scope.confined:
      val tokens = Ptr.copy(res)
      val numTokens = llama.llama_tokenize(
        ctx = ctx,
        text = Ptr.copy(prompt),
        tokens = tokens,
        n_max_tokens = res.size,
        add_bos = addBos.toByte
      )
      tokens.asArray(math.min(numTokens, ctxSize)).unsafeArray

  def decode(token: Int): String =
    ptr2str(llama.llama_token_to_str(ctx = ctx, token = token))

  def evaluate(ids: Array[Int], past: Int, params: ContextParams): Int =
    if ids.isEmpty then past
    else
      val batches = ids.grouped(params.batchSize)
      Scope.confined:
        for (batch, n) <- batches.zipWithIndex do
          llama.llama_eval(
            ctx = ctx,
            tokens = Ptr.copy(batch),
            n_tokens = batch.size,
            n_past = past + n * params.batchSize,
            n_threads = params.threads
          )
      past + ids.size

  def sample(repeatTokens: Array[Int], params: SamplingParams): Int =
    Scope.confined:
      val logits = llama.llama_get_logits(ctx).asArray(vocabSize)

      val tokenData = Array.tabulate[llama_token_data](vocabSize)(tokenId =>
        llama_token_data(id = tokenId, logit = logits(tokenId), p = 0.0)
      )
      val candidates = Ptr.copy(
        llama_token_data_array(
          data = Ptr.copy(tokenData),
          size = SizeT(tokenData.size.toShort),
          sorted = 0
        )
      )

      val repeatLastTokens = Ptr.copy(repeatTokens)
      val repeatTokensSize = SizeT(repeatTokens.size.toShort)
      llama.llama_sample_repetition_penalty(
        ctx = ctx,
        candidates = candidates,
        last_tokens = repeatLastTokens,
        last_tokens_size = repeatTokensSize,
        penalty = params.repeatPenalty
      )
      llama.llama_sample_frequency_and_presence_penalties(
        ctx = ctx,
        candidates = candidates,
        last_tokens = repeatLastTokens,
        last_tokens_size = repeatTokensSize,
        alpha_frequency = params.frequencyPenalty,
        alpha_presence = params.presencePenalty
      )

      if params.temp <= 0 then
        llama.llama_sample_token_greedy(ctx = ctx, candidates = candidates)
      else
        params.mirostat
          .collect:
            case mirostat @ Mirostat.Params(Mirostat.V1, tau, eta, m) =>
              llama.llama_sample_temperature(
                ctx = ctx,
                candidates = candidates,
                temp = params.temp
              )
              llama.llama_sample_token_mirostat(
                ctx = ctx,
                candidates = candidates,
                tau = tau,
                eta = eta,
                m = m,
                mu = Ptr.copy(mirostat.mu)
              )

            case mirostat @ Mirostat.Params(Mirostat.V2, tau, eta, _) =>
              llama.llama_sample_temperature(
                ctx = ctx,
                candidates = candidates,
                temp = params.temp
              )
              llama.llama_sample_token_mirostat_v2(
                ctx = ctx,
                candidates = candidates,
                tau = tau,
                eta = eta,
                mu = Ptr.copy(mirostat.mu)
              )
          .getOrElse:
            val topK = params.topK.filter(_ > 0).getOrElse(vocabSize)
            val minKeep = SizeT(1.toShort)
            llama.llama_sample_top_k(
              ctx = ctx,
              candidates = candidates,
              k = topK,
              min_keep = minKeep
            )
            llama.llama_sample_tail_free(
              ctx = ctx,
              candidates = candidates,
              z = params.tfsZ,
              min_keep = minKeep
            )
            llama.llama_sample_typical(
              ctx = ctx,
              candidates = candidates,
              p = params.typicalP,
              min_keep = minKeep
            )
            llama.llama_sample_top_p(
              ctx = ctx,
              candidates = candidates,
              p = params.topP,
              min_keep = minKeep
            )
            llama.llama_sample_temperature(
              ctx = ctx,
              candidates = candidates,
              temp = params.temp
            )
            llama.llama_sample_token(ctx = ctx, candidates = candidates)

  def close(suffix: Option[String]): LazyList[String] =
    llama.llama_free(ctx)
    suffix.fold(LazyList.empty)(LazyList(_))

  private def ptr2str(ptr: Ptr[Byte]): String =
    var i = 0
    while (!ptr(i) != 0) do i += 1
    String(ptr.asArray(i).unsafeArray, "UTF-8")

trait Llm(val modelPath: Path):
  def generate(prompt: String, params: LlmParams): Try[LazyList[String]]

  def close(): Unit

  def apply(prompt: String, params: LlmParams): Try[LazyList[String]] =
    generate(prompt, params)

object Llm:
  def apply(lib: Path, model: Path, params: ContextParams): Llm =
    val binding = Try:
      System.load(lib.toAbsolutePath.toString)
      FSet.instance[Llama]

    binding.foreach(_.llama_init_backend())

    val defaultParams = binding.map: llama =>
      llama.llama_context_default_params()

    val llm = Scope.confined:
      val baseModel = for
        llama <- binding
        defaultParams <- defaultParams
      yield llama.llama_load_model_from_file(
        path_model = Ptr.copy(model.toAbsolutePath.toString),
        params = llamaParams(defaultParams, params)
      )

      params.lora.adapter.fold(baseModel)(loraAdapter =>
        val err =
          for
            llama <- binding
            llm <- baseModel
          yield llama.llama_model_apply_lora_from_file(
            model = llm,
            path_lora = Ptr.copy(loraAdapter.toAbsolutePath.toString),
            path_base_model =
              Ptr.copy(params.lora.base.fold("")(_.toAbsolutePath.toString)),
            n_threads = params.threads
          )
        err.filter(_ == 0).flatMap(_ => baseModel)
      )

    new Llm(model) {
      def generate(prompt: String, params: LlmParams): Try[LazyList[String]] =
        Scope.confined:
          val ctx = for
            llama <- binding
            llm <- llm
            defaultParams <- defaultParams
          yield llama.llama_new_context_with_model(
            model = llm,
            params = llamaParams(defaultParams, params.context)
          )
          ctx.filter(_ != null).map(new SlincLlm(_).generate(prompt, params))

      def close(): Unit =
        for
          llama <- binding
          llm <- llm
        do llama.llama_free_model(llm)
    }

  private def llamaParams(
      defaultParams: llama_context_params,
      params: ContextParams
  ): llama_context_params =
    defaultParams.copy(
      seed = params.seed,
      n_ctx = params.contextSize,
      n_batch = params.batchSize,
      n_gpu_layers = params.gpuLayers,
      main_gpu = params.mainGpu,
      low_vram = if params.lowVram then 1 else 0,
      f16_kv = if params.f16 then 1 else 0,
      logits_all = 0,
      vocab_only = 0,
      use_mmap = if params.mmap then 1 else 0,
      use_mlock = if params.mlock then 1 else 0,
      embedding = 0
    )
