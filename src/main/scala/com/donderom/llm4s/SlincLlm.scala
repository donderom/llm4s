package com.donderom.llm4s

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

import scala.collection.mutable.ArrayDeque

import fr.hammons.slinc.runtime.given
import fr.hammons.slinc.types.SizeT
import fr.hammons.slinc.{FSet, Ptr, Scope}

private class SlincLlm private[llm4s] (private[llm4s] val ctx: Ptr[Any]):
  import State.*

  final case class Sample(id: Int, prob: Option[Probability])

  val llama = FSet.instance[Llama]

  lazy val decoder = StandardCharsets.UTF_8.newDecoder

  def generate(prompt: String, params: LlmParams): LazyList[Token] =
    val lastTokens = new ArrayDeque[Int](ctxSize)
    val stop = Stop.Acc[Token](params.stopSeqs)

    def eval(start: Int, past: Evaluated): Evaluated =
      val ids = lastTokens.slice(start, lastTokens.size).toArray
      evaluate(ids, past, params.context)

    def repeatTokens(): Array[Int] =
      val repeatLastTokens =
        if params.sampling.repeatLastTokens < 0 then ctxSize
        else params.sampling.repeatLastTokens
      val lastRepeat = math.min(lastTokens.size, repeatLastTokens)
      lastTokens.takeRight(lastRepeat).toArray

    def tokens(state: State[Token]): LazyList[Token] =
      if !state.remaining.none then
        val past = state.evaluated
        val newPast = if past.incr.toInt > ctxSize then
          val start = ctxSize - ((past.toInt - params.keepTokens) / 2) - 1
          eval(start, Evaluated(math.max(1, params.keepTokens)))
        else
          val start =
            if lastTokens.size == ctxSize then ctxSize - 1 else past.toInt
          eval(start, past)

        val smpl = sample(repeatTokens(), params.sampling, params.logitBias)

        if lastTokens.size == ctxSize then lastTokens.remove(0)
        lastTokens.append(smpl.id)

        if lastTokens.lastOption.fold(true)(_ != eosToken) then
          decode(smpl.id, state.partialBytes) match
            case partial: Array[Byte] =>
              tokens(state.partial(newPast, partial, smpl.prob))
            case token: String =>
              val probs = (state.probs :+ smpl.prob).flatten
              val gen = (stop: Stop.State[Token]) =>
                tokens(state.regular(newPast, stop))
              stop.step(Token(token, probs), state.stop) match
                case stop.Action.Cont(st) => gen(st)
                case stop.Action.Emit(chunk: Token, st) =>
                  chunk #:: gen(st)
                case stop.Action.Emit(chunk: Vector[Token], st) =>
                  LazyList.from(chunk) #::: gen(st)
                case stop.Action.Stop(chunk) =>
                  LazyList.from(params.suffix.fold(chunk)(chunk :+ _.token))
        else close(state.stop.deferred(params.suffix))
      else close(state.stop.deferred(params.suffix))
    end tokens

    val ids = encode(prompt)
    ids.foreach(lastTokens.append)
    val gen = (e: Evaluated) => tokens(State[Token](params.predictTokens, e))
    if params.echo then promptTokens(ids, Array()) #::: gen(Evaluated.none)
    else gen(evaluate(ids, Evaluated.none, params.context))
  end generate

  def promptTokens(ids: Array[Int], pending: Array[Byte]): LazyList[Token] =
    if ids.isEmpty then LazyList.empty
    else
      decode(ids.head, pending) match
        case token: String => Token(token) #:: promptTokens(ids.tail, Array())
        case partial: Array[Byte] => promptTokens(ids.tail, partial)

  def embeddings(prompt: String, params: ContextParams): Array[Float] =
    val ids = encode(prompt)
    val _ = evaluate(ids, Evaluated.none, params)
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

  def decode(token: Int): String | Array[Byte] = decode(token, Array())

  def decode(token: Int, pending: Array[Byte]): String | Array[Byte] =
    val tokenPtr = llama.llama_token_to_str(ctx = ctx, token = token)
    var i = 0
    while (!tokenPtr(i) != 0) do i += 1
    val bytes = Array.concat(pending, tokenPtr.asArray(i).unsafeArray)
    try decoder.decode(ByteBuffer.wrap(bytes)).toString
    catch case _ => bytes

  def evaluate(
      ids: Array[Int],
      past: Evaluated,
      params: ContextParams
  ): Evaluated =
    if ids.isEmpty then past
    else
      val batches = ids.grouped(params.batchSize)
      Scope.confined:
        for (batch, n) <- batches.zipWithIndex do
          llama.llama_eval(
            ctx = ctx,
            tokens = Ptr.copy(batch),
            n_tokens = batch.size,
            n_past = (past + n * params.batchSize).toInt,
            n_threads = params.threads
          )
      past + ids.size

  def sample(
      repeatTokens: Array[Int],
      params: SamplingParams,
      logitBias: Map[Int, Float]
  ): Sample =
    Scope.confined:
      val logits = llama.llama_get_logits(ctx).asArray(vocabSize).unsafeArray
      logitBias.foreach((token, bias) => logits(token) = bias)

      val tokenData = Array.tabulate[llama_token_data](vocabSize): tokenId =>
        llama_token_data(id = tokenId, logit = logits(tokenId), p = .0)

      val data = Ptr.copy(tokenData)

      val candidates = Ptr.copy(
        llama_token_data_array(
          data = data,
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

      val tokenId = if params.temp <= 0 then
        val id = llama.llama_sample_token_greedy(ctx, candidates)
        if params.logprobs > 0 then llama.llama_sample_softmax(ctx, candidates)
        id
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
          val minKeep = SizeT(math.max(1, params.logprobs).toShort)
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
      Sample(tokenId, logprob(tokenId, data, params.logprobs))
  end sample

  def logprob(
      id: Int,
      data: Ptr[llama_token_data],
      num: Int
  ): Option[Probability] =
    def tokenValue(tokenId: Int): String =
      decode(tokenId) match
        case token: String => token
        case bytes: Array[Byte] =>
          bytes.map(b => s"\\\\x${String.format("%02x", b)}").mkString

    if num > 0 then
      val log = (td: llama_token_data) => math.log(td.p)
      val cap = math.min(num, vocabSize)
      val logprobs = data.asArray(cap).unsafeArray.map: td =>
        Logprob(tokenValue(td.id), log(td))
      val current = LazyList.range(0, vocabSize).map(!data(_)).find(_.id == id)
      val logprob = Logprob(tokenValue(id), current.fold(.0)(log))
      Some(Probability(logprob, logprobs))
    else None

  def close(suffix: Vector[Token]): LazyList[Token] =
    llama.llama_free(ctx)
    LazyList.from(suffix)
