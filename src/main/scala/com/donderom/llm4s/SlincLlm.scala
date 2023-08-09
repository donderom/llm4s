package com.donderom.llm4s

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

import scala.collection.mutable.ArrayDeque

import fr.hammons.slinc.runtime.given
import fr.hammons.slinc.types.SizeT
import fr.hammons.slinc.{FSet, Ptr, Scope}

private class SlincLlm private[llm4s] (private[llm4s] val ctx: Ptr[Any]):
  val llama = FSet.instance[Llama]

  lazy val decoder = StandardCharsets.UTF_8.newDecoder

  def generate(prompt: String, params: LlmParams): LazyList[String] =
    val lastTokens = new ArrayDeque[Int](ctxSize)
    val stop = Stop.Acc(params.stopSeqs)

    def tokens(
        remaining: Int,
        evaluated: Int,
        partialBytes: Array[Byte],
        state: Stop.State
    ): LazyList[String] =
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
          decode(id, partialBytes) match
            case partial: Array[Byte] =>
              tokens(remaining - 1, newPast, partial, state)
            case token: String =>
              val gen = tokens(remaining - 1, newPast, Array(), _)
              stop.step(token, state) match
                case Stop.Action.Cont(state) => gen(state)
                case Stop.Action.Emit(chunk: String, state) =>
                  chunk #:: gen(state)
                case Stop.Action.Emit(chunk: Vector[String], state) =>
                  LazyList.from(chunk) #::: gen(state)
                case Stop.Action.Stop(chunk) =>
                  LazyList.from(params.suffix.fold(chunk)(chunk :+ _))
        else close(state.deferred(params.suffix))
      else close(state.deferred(params.suffix))
    end tokens

    val ids = encode(prompt)
    ids.foreach(lastTokens.append)
    val gen = tokens(params.predictTokens, _, Array(), Stop.State())
    if params.echo then promptTokens(ids, Array()) #::: gen(0)
    else gen(evaluate(ids = ids, past = 0, params = params.context))
  end generate

  def promptTokens(ids: Array[Int], pending: Array[Byte]): LazyList[String] =
    if ids.isEmpty then LazyList.empty
    else
      decode(ids.head, pending) match
        case token: String        => token #:: promptTokens(ids.tail, Array())
        case partial: Array[Byte] => promptTokens(ids.tail, partial)

  def embeddings(prompt: String, params: ContextParams): Array[Float] =
    val ids = encode(prompt)
    val _ = evaluate(ids, 0, params)
    val size = llama.llama_n_embd(ctx)
    val embeddings = llama.llama_get_embeddings(ctx).asArray(size).unsafeArray
    llama.llama_free(ctx)
    embeddings

  lazy val ctxSize: Int = llama.llama_n_ctx(ctx)
  lazy val eosToken: Int = llama.llama_token_eos()
  lazy val vocabSize: Int = llama.llama_n_vocab(ctx)

  def encode(prompt: String): Array[Int] = encode(" " + prompt, true)

  def encode(text: String, addBos: Boolean): Array[Int] =
    val bos = addBos.toByte
    val bytes = text.getBytes(StandardCharsets.UTF_8)
    val res = new Array[Int](bytes.size + bos)
    Scope.confined:
      val tokens = Ptr.copy(res)
      val numTokens = llama.llama_tokenize(
        ctx = ctx,
        text = Ptr.copy(bytes),
        tokens = tokens,
        n_max_tokens = res.size,
        add_bos = bos
      )
      tokens.asArray(math.min(numTokens, ctxSize)).unsafeArray

  def decode(token: Int, pending: Array[Byte]): String | Array[Byte] =
    val tokenPtr = llama.llama_token_to_str(ctx = ctx, token = token)
    var i = 0
    while (!tokenPtr(i) != 0) do i += 1
    val bytes = Array.concat(pending, tokenPtr.asArray(i).unsafeArray)
    try decoder.decode(ByteBuffer.wrap(bytes)).toString
    catch case _ => bytes

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

      val tokenData = Array.tabulate[llama_token_data](vocabSize): tokenId =>
        llama_token_data(id = tokenId, logit = logits(tokenId), p = 0.0)

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
        params.mirostat.collect:
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
  end sample

  def close(suffix: Vector[String]): LazyList[String] =
    llama.llama_free(ctx)
    LazyList.from(suffix)
