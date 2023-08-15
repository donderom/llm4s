package com.donderom.llm4s

import java.nio.file.Path

final case class LoraParams(
    adapter: Option[Path] = None,
    base: Option[Path] = None
)

final case class RopeParams(freqBase: Float = 10000.0f, freqScale: Float = 1.0f)

final case class ContextParams(
    contextSize: Int = 512,
    batchSize: Int = 512,
    gpuLayers: Int = 0,
    mainGpu: Int = 0,
    lowVram: Boolean = false,
    seed: Int = -1,
    f16: Boolean = true,
    mmap: Boolean = true,
    mlock: Boolean = false,
    threads: Int = 1,
    lora: LoraParams = LoraParams(),
    numa: Boolean = false,
    rope: RopeParams = RopeParams()
)

final case class Penalty(
    repeat: Float = 1.10f,
    frequency: Float = .0f,
    presence: Float = .0f
)

object Default:
  val penalty: Penalty = Penalty()
  val repeatLastTokens: Int = 64
  val logprobs: Int = 0
  val temp: Float = .8f
  object Mirostat:
    val tau: Float = 5.0f
    val eta: Float = .1f
    val muCoef: Float = 2.0f

enum Sampling(
    val penalty: Penalty,
    val repeatLastTokens: Int,
    val logprobs: Int
):
  case Greedy(
      override val penalty: Penalty = Default.penalty,
      override val repeatLastTokens: Int = Default.repeatLastTokens,
      override val logprobs: Int = Default.logprobs
  ) extends Sampling(penalty, repeatLastTokens, logprobs)

  case MirostatV1(
      override val penalty: Penalty = Default.penalty,
      override val repeatLastTokens: Int = Default.repeatLastTokens,
      override val logprobs: Int = Default.logprobs,
      temp: Float = Default.temp,
      tau: Float = Default.Mirostat.tau,
      eta: Float = Default.Mirostat.eta,
      m: Int = 100,
      muCoef: Float = Default.Mirostat.muCoef
  ) extends Sampling(penalty, repeatLastTokens, logprobs)

  case MirostatV2(
      override val penalty: Penalty = Default.penalty,
      override val repeatLastTokens: Int = Default.repeatLastTokens,
      override val logprobs: Int = Default.logprobs,
      temp: Float = Default.temp,
      tau: Float = Default.Mirostat.tau,
      eta: Float = Default.Mirostat.eta,
      muCoef: Float = Default.Mirostat.muCoef
  ) extends Sampling(penalty, repeatLastTokens, logprobs)

  case Random(
      override val penalty: Penalty = Default.penalty,
      override val repeatLastTokens: Int = Default.repeatLastTokens,
      override val logprobs: Int = Default.logprobs,
      temp: Float = Default.temp,
      topK: Option[Int] = Some(40),
      tfsZ: Float = 1.0f,
      typicalP: Float = 1.0f,
      topP: Float = .95f
  ) extends Sampling(penalty, repeatLastTokens, logprobs)

final case class LlmParams(
    context: ContextParams = ContextParams(),
    sampling: Sampling = Sampling.Random(),
    predictTokens: Int = -1,
    keepTokens: Int = 0,
    logitBias: Map[Int, Float] = Map(),
    suffix: Option[String] = None,
    echo: Boolean = true,
    stopSeqs: List[String] = Nil
)
