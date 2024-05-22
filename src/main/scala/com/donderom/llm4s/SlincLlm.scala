package com.donderom.llm4s

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

import scala.collection.mutable.ArrayDeque

import fr.hammons.slinc.runtime.given
import fr.hammons.slinc.types.SizeT
import fr.hammons.slinc.{FSet, Ptr, Scope}

import State.*

private class SlincLlm private[llm4s] (private[llm4s] val ctx: Llama.Ctx):
  final case class Sample(id: Int, prob: Option[Probability])

  lazy val llama = FSet.instance[Llama]

  lazy val model = llama.llama_get_model(ctx)
  lazy val decoder = StandardCharsets.UTF_8.newDecoder

  def generate(prompt: String, params: LlmParams): Usage =
    val lastTokens = new ArrayDeque[Int](ctxSize)
    val stop = Stop.Acc[Token](params.stopSeqs)

    def eval(evaluated: Evaluated): Evaluated =
      val past = if evaluated.incr.toInt > ctxSize then
        val keepTokens = params.keepTokens + (if addBos then 1 else 0)
        val left = evaluated.toInt - keepTokens
        val discard = left / 2
        llama.llama_kv_cache_seq_rm(
          ctx = ctx,
          seq_id = 0,
          p0 = keepTokens,
          p1 = keepTokens + discard
        )
        llama.llama_kv_cache_seq_add(
          ctx = ctx,
          seq_id = 0,
          p0 = keepTokens + discard,
          p1 = evaluated.toInt,
          delta = -discard
        )
        evaluated - discard
      else evaluated

      val start =
        if lastTokens.size == ctxSize then ctxSize - 1 else evaluated.toInt
      val ids = lastTokens.slice(start, lastTokens.size).toArray
      evaluate(ids, past, params.context.batch)

    def repeatTokens(): Array[Int] =
      val repeatLastTokens =
        if params.sampling.repeatLastTokens < 0 then ctxSize
        else params.sampling.repeatLastTokens
      val lastRepeat = math.min(lastTokens.size, repeatLastTokens)
      val padding = Array.fill(repeatLastTokens - lastRepeat)(0)
      padding ++ lastTokens.takeRight(lastRepeat).toArray

    def tokens(state: State[Token]): LazyList[Token] =
      if !state.remaining.none then
        val newPast = eval(state.evaluated)
        val smpl = sample(repeatTokens(), params.sampling, params.logitBias)

        if lastTokens.size == ctxSize then lastTokens.remove(0)
        lastTokens.append(smpl.id)

        if lastTokens.lastOption.fold(true)(keepGenerating) then
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
    Usage(
      ids.size,
      if params.echo then promptTokens(ids, Array()) #::: gen(Evaluated.none)
      else gen(evaluate(ids, Evaluated.none, params.context.batch))
    )
  end generate

  def promptTokens(ids: Array[Int], pending: Array[Byte]): LazyList[Token] =
    if ids.isEmpty then LazyList.empty
    else
      decode(ids.head, pending) match
        case token: String => Token(token) #:: promptTokens(ids.tail, Array())
        case partial: Array[Byte] => promptTokens(ids.tail, partial)

  def embeddings(prompt: String, params: BatchParams): Array[Float] =
    val ids = encode(prompt)
    val _ = evaluate(ids, Evaluated.none, params)
    val size = llama.llama_n_embd(model)
    val embeddings = llama.llama_get_embeddings(ctx).asArray(size).unsafeArray
    llama.llama_free(ctx)
    embeddings

  lazy val ctxSize: Int = llama.llama_n_ctx(ctx)
  lazy val vocabSize: Int = llama.llama_n_vocab(model)
  lazy val addBosToken: Int = llama.llama_add_bos_token(model)
  lazy val addBos: Boolean =
    if addBosToken != -1 then addBosToken != 0
    else llama.llama_vocab_type(model) == Llama.VocabType.LLAMA_VOCAB_TYPE_SPM

  def keepGenerating(token: Int): Boolean =
    !llama.llama_token_is_eog(model, token)

  def encode(text: String): Array[Int] =
    val bos = if addBos then 1 else 0
    val bytes = text.getBytes(StandardCharsets.UTF_8)
    val res = new Array[Int](bytes.size + bos)
    Scope.confined:
      val tokens = Ptr.copy(res)
      val numTokens = llama.llama_tokenize(
        model = model,
        text = Ptr.copy(bytes),
        text_len = bytes.size,
        tokens = tokens,
        n_tokens_max = res.size,
        add_special = addBos,
        parse_special = true
      )
      tokens.asArray(math.min(numTokens, ctxSize)).unsafeArray

  val pieceLength = 8

  def decode(token: Int): String | Array[Byte] = decode(token, Array())

  def decode(token: Int, pending: Array[Byte]): String | Array[Byte] =
    decode(token, pending, pieceLength)

  def decode(token: Int, pending: Array[Byte], len: Int): String | Array[Byte] =
    val res = new Array[Byte](len)
    Scope.confined:
      val tokens = Ptr.copy(res)
      val numTokens = llama.llama_token_to_piece(
        model = model,
        token = token,
        buf = tokens,
        length = res.size,
        special = false
      )
      if numTokens < 0 then decode(token, pending, math.abs(numTokens))
      else
        val bytes = Array.concat(pending, tokens.asArray(numTokens).unsafeArray)
        try decoder.decode(ByteBuffer.wrap(bytes)).toString
        catch case _ => bytes

  def evaluate(
      ids: Array[Int],
      past: Evaluated,
      params: BatchParams
  ): Evaluated =
    if ids.isEmpty then past
    else
      val batches = ids.grouped(params.size)
      Scope.confined:
        for (batch, n) <- batches.zipWithIndex do
          llama.llama_decode(
            ctx = ctx,
            batch = llama.llama_batch_get_one(
              tokens = Ptr.copy(batch),
              n_tokens = batch.size,
              pos_0 = (past + n * params.size).toInt,
              seq_id = 0
            )
          )
      past + ids.size

  def sample(
      repeatTokens: Array[Int],
      sampling: Sampling,
      logitBias: Map[Int, Float],
      idx: Int = 0
  ): Sample =
    import Sampling.*

    Scope.confined:
      val logits = llama.llama_get_logits_ith(ctx, idx).asArray(vocabSize)
        .unsafeArray
      logitBias.foreach((token, bias) => logits(token) = bias)

      val tokenData = Array.tabulate[Llama.TokenData](vocabSize): tokenId =>
        Llama.TokenData(id = tokenId, logit = logits(tokenId), p = .0)

      val data = Ptr.copy(tokenData)

      val candidates = Ptr.copy(
        Llama.TokenDataArray(
          data = data,
          size = SizeT(tokenData.size.toShort),
          sorted = false
        )
      )

      val repeatLastTokens = Ptr.copy(repeatTokens)
      val repeatTokensSize = SizeT(repeatTokens.size.toShort)
      llama.llama_sample_repetition_penalties(
        ctx = ctx,
        candidates = candidates,
        last_tokens = repeatLastTokens,
        penalty_last_n = repeatTokensSize,
        penalty_repeat = sampling.penalty.repeat,
        penalty_freq = sampling.penalty.frequency,
        penalty_present = sampling.penalty.presence
      )

      val tokenId = sampling match
        case Greedy(_, _, logprobs) =>
          if logprobs > 0 then
            llama.llama_sample_softmax(ctx, candidates)
            (!data).id
          else llama.llama_sample_token_greedy(ctx, candidates)

        case MirostatV1(_, _, _, temp, tau, eta, m, muCoef) =>
          llama.llama_sample_temp(ctx, candidates, temp)
          llama.llama_sample_token_mirostat(
            ctx = ctx,
            candidates = candidates,
            tau = tau,
            eta = eta,
            m = m,
            mu = Ptr.copy(muCoef * tau)
          )

        case MirostatV2(_, _, _, temp, tau, eta, muCoef) =>
          llama.llama_sample_temp(ctx, candidates, temp)
          llama.llama_sample_token_mirostat_v2(
            ctx = ctx,
            candidates = candidates,
            tau = tau,
            eta = eta,
            mu = Ptr.copy(muCoef * tau)
          )

        case Random(
              _,
              _,
              logprobs,
              temp,
              topK,
              tfsZ,
              typicalP,
              topP,
              minP,
              dynatemp,
              samplers
            ) =>
          val topk = topK.filter(_ > 0).getOrElse(vocabSize)
          val minKeep = SizeT(math.max(1, logprobs).toShort)
          samplers.foreach:
            case Sampler.TOP_K =>
              llama.llama_sample_top_k(ctx, candidates, topk, minKeep)
            case Sampler.TAIL_FREE =>
              llama.llama_sample_tail_free(ctx, candidates, tfsZ, minKeep)
            case Sampler.TYPICAL =>
              llama.llama_sample_typical(ctx, candidates, typicalP, minKeep)
            case Sampler.TOP_P =>
              llama.llama_sample_top_p(ctx, candidates, topP, minKeep)
            case Sampler.MIN_P =>
              llama.llama_sample_min_p(ctx, candidates, minP, minKeep)
            case Sampler.TEMPERATURE =>
              if dynatemp.range > 0 then
                val dynatemp_min = math.max(.0f, temp - dynatemp.range)
                val dynatemp_max = math.max(.0f, temp + dynatemp.range)
                llama.llama_sample_entropy(
                  ctx = ctx,
                  candidates_p = candidates,
                  min_temp = dynatemp_min,
                  max_temp = dynatemp_max,
                  exponent_val = dynatemp.exponent
                )
              else llama.llama_sample_temp(ctx, candidates, temp)
          llama.llama_sample_token(ctx, candidates)

      Sample(tokenId, logprob(tokenId, data, sampling.logprobs))
  end sample

  def logprob(
      id: Int,
      data: Ptr[Llama.TokenData],
      num: Int
  ): Option[Probability] =
    def tokenValue(tokenId: Int): String =
      decode(tokenId) match
        case token: String => token
        case bytes: Array[Byte] =>
          bytes.map(b => s"\\\\x${String.format("%02x", b)}").mkString

    if num > 0 then
      val log = (td: Llama.TokenData) => math.log(td.p)
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
