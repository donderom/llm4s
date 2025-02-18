package com.donderom.llm4s

import fr.hammons.slinc.runtime.given
import fr.hammons.slinc.types.SizeT
import fr.hammons.slinc.{FSet, Ptr, Scope}

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

import scala.collection.mutable.ArrayDeque

import State.*

private class SlincLlm private[llm4s] (private[llm4s] val ctx: Llama.Ctx):
  // Logprobs are None until a better solution is implemented
  final case class Sample(id: Int, prob: Option[Probability])

  lazy val llama = FSet.instance[Llama]

  lazy val model = llama.llama_get_model(ctx)
  lazy val decoder = StandardCharsets.UTF_8.newDecoder

  def generate(prompt: String, params: LlmParams): Usage =
    val sampler = createSampler(params.sampling)
    val lastTokens = new ArrayDeque[Int](ctxSize)
    val stop = Stop.Acc[Token](params.stopSeqs)
    val ids = encode(prompt)
    val keepTokens =
      if params.keepTokens < 0 || params.keepTokens > ids.size then ids.size
      else params.keepTokens + (if addBos then 1 else 0)

    def eval(evaluated: Evaluated): Evaluated =
      val past =
        if evaluated.incr.toInt > ctxSize then
          if params.groupAttention.factor == 1 then
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
          else
            val factor = params.groupAttention.factor
            val width = params.groupAttention.width

            @annotation.tailrec
            def selfExtend(past: Evaluated, kvTokens: Int): Evaluated =
              if past.toInt >= kvTokens + width then
                val ib = (factor * kvTokens) / width
                val bd = (width / factor) * (factor - 1)
                val dd = (width / factor) - ib * bd - width
                llama.llama_kv_cache_seq_add(
                  ctx = ctx,
                  seq_id = 0,
                  p0 = kvTokens,
                  p1 = past.toInt,
                  delta = ib * bd
                )
                llama.llama_kv_cache_seq_div(
                  ctx = ctx,
                  seq_id = 0,
                  p0 = kvTokens + ib * bd,
                  p1 = kvTokens + ib * bd + width,
                  d = factor
                )
                llama.llama_kv_cache_seq_add(
                  ctx = ctx,
                  seq_id = 0,
                  p0 = kvTokens + ib * bd + width,
                  p1 = past.toInt + ib * bd,
                  delta = dd
                )
                selfExtend(past - bd, width / factor)
              else past

            selfExtend(evaluated, 0)
        else evaluated

      val start =
        if lastTokens.size == ctxSize then ctxSize - 1 else evaluated.toInt
      val ids = lastTokens.slice(start, lastTokens.size).toArray
      evaluate(ids, past, params.context.batch)
    end eval

    def tokens(state: State[Token]): LazyList[Token] =
      if !state.remaining.none then
        val newPast = eval(state.evaluated)

        val tokenId = llama.llama_sampler_sample(sampler, ctx, -1)
        llama.llama_sampler_accept(sampler, tokenId)
        val smpl = Sample(tokenId, None)

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
        else close(state.stop.deferred(params.suffix), sampler)
      else close(state.stop.deferred(params.suffix), sampler)
    end tokens

    ids.foreach(lastTokens.append)

    // Support encoder-decoder models
    val encoder = llama.llama_model_has_encoder(model)
    if encoder then
      Scope.confined:
        llama.llama_encode(
          ctx = ctx,
          batch = llama.llama_batch_get_one(Ptr.copy(ids), ids.size)
        )
      val decStartToken = llama.llama_model_decoder_start_token(model)
      if !nullToken(decStartToken) then lastTokens.append(decStartToken)
      else lastTokens.append(llama.llama_vocab_bos(vocab))

    val gen = (e: Evaluated) => tokens(State[Token](params.predictTokens, e))
    Usage(
      ids.size,
      if encoder then gen(Evaluated(ids.size))
      else if params.echo then promptTokens(ids) #::: gen(Evaluated.none)
      else gen(evaluate(ids, Evaluated.none, params.context.batch))
    )
  end generate

  def promptTokens(ids: Array[Int]): LazyList[Token] =
    promptTokens(ids, Array())

  def promptTokens(ids: Array[Int], pending: Array[Byte]): LazyList[Token] =
    if ids.isEmpty then LazyList.empty
    else
      decode(ids.head, pending) match
        case token: String        => Token(token) #:: promptTokens(ids.tail)
        case partial: Array[Byte] => promptTokens(ids.tail, partial)

  def embeddings(prompt: String, params: EmbeddingParams): Array[Float] =
    val ids = encode(prompt)
    val _ = evaluate(ids, Evaluated.none, params.context.batch)
    val size = llama.llama_model_n_embd(model)
    val embeddings = llama.llama_get_embeddings(ctx).asArray(size).unsafeArray
    llama.llama_free(ctx)

    def normalized(
        f: (Float, Float) => Float,
        post: Float => Float = identity
    ) =
      val sum = post(embeddings.foldLeft(0.0f)(f))
      val norm = if sum > 0.0f then 1.0f / sum else 0.0f
      embeddings.map(_ * norm)

    params.norm match
      case Some(Norm.MaxAbsolute) =>
        normalized(
          (sum, emb) =>
            val absEmb = math.abs(emb)
            if sum < absEmb then absEmb else sum
          ,
          _ / 32760.0f
        )
      case Some(Norm.Taxicab) =>
        normalized((sum, emb) => sum + math.abs(emb))
      case Some(Norm.Euclidean) =>
        normalized(
          (sum, emb) => sum + emb * emb,
          sum => math.sqrt(sum).toFloat
        )
      case Some(Norm.PNorm(p)) =>
        normalized(
          (sum, emb) => sum + math.pow(math.abs(emb), p).toFloat,
          sum => math.pow(sum, 1.0 / p).toFloat
        )
      case _ => embeddings
  end embeddings

  lazy val ctxSize: Int = llama.llama_n_ctx(ctx)
  lazy val vocab: Llama.Vocab = llama.llama_model_get_vocab(model)
  lazy val vocabSize: Int = llama.llama_vocab_n_tokens(vocab)
  lazy val addBos: Boolean = llama.llama_vocab_get_add_bos(vocab)

  def nullToken(token: Int): Boolean = token == Llama.nullToken

  def keepGenerating(token: Int): Boolean =
    !llama.llama_vocab_is_eog(vocab, token)

  // Reserve capacity for BOS and EOS tokens (used by tokenize)
  val specialTokensNum = 2

  def encode(text: String): Array[Int] =
    val bytes = text.getBytes(StandardCharsets.UTF_8)
    val res = new Array[Int](bytes.size + specialTokensNum)
    Scope.confined:
      val tokens = Ptr.copy(res)
      val numTokens = llama.llama_tokenize(
        vocab = vocab,
        text = Ptr.copy(bytes),
        text_len = bytes.size,
        tokens = tokens,
        n_tokens_max = res.size,
        add_special = true,
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
        vocab = vocab,
        token = token,
        buf = tokens,
        length = res.size,
        lstrip = 0,
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
      batch: BatchParams
  ): Evaluated =
    if ids.isEmpty then past
    else
      val batches = ids.grouped(batch.logical)
      Scope.confined:
        for (batch, n) <- batches.zipWithIndex do
          llama.llama_decode(
            ctx = ctx,
            batch = llama.llama_batch_get_one(
              tokens = Ptr.copy(batch),
              n_tokens = batch.size
            )
          )
      past + ids.size

  def createSampler(params: Sampling): Llama.Sampler =
    val sparams = llama.llama_sampler_chain_default_params()
    val chain = llama.llama_sampler_chain_init(sparams)
    val add = llama.llama_sampler_chain_add(chain, _)
    params match
      case config: Sampling.Dist =>
        Scope.confined:
          if !config.logitBias.isEmpty then
            val logitBias = config.logitBias.map(Llama.LogitBias(_, _))
            add(
              llama.llama_sampler_init_logit_bias(
                vocabSize,
                config.logitBias.size,
                Ptr.copy(logitBias.toArray)
              )
            )

          for sampler <- config.samplers do
            val minKeep = SizeT(config.minKeep)
            sampler match
              case SamplerType.DRY =>
                val seqBreakers = config.dry.seqBreakers.map(_.toByte)
                add(
                  llama.llama_sampler_init_dry(
                    llama.llama_model_get_vocab(model),
                    llama.llama_model_n_ctx_train(model),
                    config.dry.multiplier,
                    config.dry.base,
                    config.dry.allowedLength,
                    config.dry.penaltyLastN,
                    Ptr.copy(Ptr.copy(seqBreakers.toArray)),
                    SizeT(seqBreakers.size.toShort)
                  )
                )

              case SamplerType.TOP_K =>
                add(llama.llama_sampler_init_top_k(config.topK))

              case SamplerType.TOP_P =>
                add(llama.llama_sampler_init_top_p(config.topP, minKeep))

              case SamplerType.MIN_P =>
                add(llama.llama_sampler_init_min_p(config.minP, minKeep))

              case SamplerType.XTC =>
                add(
                  llama.llama_sampler_init_xtc(
                    config.xtc.probability,
                    config.xtc.threshold,
                    minKeep,
                    config.seed
                  )
                )

              case SamplerType.TYPICAL_P =>
                add(
                  llama.llama_sampler_init_typical(
                    config.typicalP,
                    minKeep
                  )
                )

              case SamplerType.TEMPERATURE =>
                add(
                  llama.llama_sampler_init_temp_ext(
                    config.temp,
                    config.dynatemp.range,
                    config.dynatemp.exponent
                  )
                )

              case SamplerType.PENALTIES =>
                add(
                  llama.llama_sampler_init_penalties(
                    config.penalty.lastN,
                    config.penalty.repeat,
                    config.penalty.frequency,
                    config.penalty.presence
                  )
                )

          if config.greedy then add(llama.llama_sampler_init_greedy())
          else add(llama.llama_sampler_init_dist(config.seed))

      case Sampling.Mirostat1(seed, temp, tau, eta, m) =>
        add(llama.llama_sampler_init_temp(temp))
        add(llama.llama_sampler_init_mirostat(vocabSize, seed, tau, eta, m))

      case Sampling.Mirostat2(seed, temp, tau, eta) =>
        add(llama.llama_sampler_init_temp(temp))
        add(llama.llama_sampler_init_mirostat_v2(seed, tau, eta))

    chain
  end createSampler

  def close(suffix: Vector[Token], sampler: Llama.Sampler): LazyList[Token] =
    llama.llama_sampler_free(sampler)
    llama.llama_free(ctx)
    LazyList.from(suffix)
