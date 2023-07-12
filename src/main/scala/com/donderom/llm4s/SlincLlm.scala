package com.donderom.llm4s

import scala.collection.mutable.ArrayDeque

import fr.hammons.slinc.runtime.given
import fr.hammons.slinc.types.SizeT
import fr.hammons.slinc.{FSet, Ptr, Scope}

private class SlincLlm private[llm4s] (private[llm4s] val ctx: Ptr[Any]):
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
        val id = sample(repeatTokens, params.sampling, params.logitBias)

        if lastTokens.size == ctxSize then lastTokens.remove(0)
        lastTokens.append(id)

        if lastTokens.lastOption.fold(true)(_ != eosToken) then
          decode(id) #:: tokens(remaining = remaining - 1, evaluated = newPast)
        else close(params.suffix)
      else close(params.suffix)

    val ids = encode(" " + prompt)
    if params.echo then
      val promptTokens =
        LazyList.from(ids).tapEach(lastTokens.append).map(decode)
      promptTokens #::: tokens(remaining = params.predictTokens, evaluated = 0)
    else
      ids.foreach(lastTokens.append)
      val evaluated = evaluate(ids = ids, past = 0, params = params.context)
      tokens(remaining = params.predictTokens, evaluated = evaluated)

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

  def sample(
      repeatTokens: Array[Int],
      params: SamplingParams,
      logitBias: Map[Int, Float]
  ): Int =
    Scope.confined:
      val logits = llama.llama_get_logits(ctx).asArray(vocabSize).unsafeArray
      logitBias.foreach((token, bias) => logits(token) = bias)

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
