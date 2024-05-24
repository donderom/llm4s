package com.donderom.llm4s

import java.nio.file.Path

import Llama.{NumaStrategy, RopeScalingType}

object Default:
  val threads = Runtime.getRuntime.availableProcessors

  val penalty: Penalty = Penalty()
  val repeatLastTokens: Int = 64
  val logprobs: Int = 0
  val temp: Float = .8f
  object Mirostat:
    val tau: Float = 5.0f
    val eta: Float = .1f
    val muCoef: Float = 2.0f

final case class LoraParams(
    adapter: Option[Path] = None,
    base: Option[Path] = None,
    scale: Float = 1.0f,
    threads: Int = Default.threads
)

final case class ModelParams(
    gpuLayers: Int = 0,
    mainGpu: Int = 0,
    mmap: Boolean = true,
    mlock: Boolean = false,
    numa: NumaStrategy = NumaStrategy.DISABLED,
    lora: LoraParams = LoraParams()
)

final case class RopeParams(
    scalingType: RopeScalingType = RopeScalingType.UNSPECIFIED,
    freqBase: Float = 0.0f,
    freqScale: Float = 0.0f
)

final case class YarnParams(
    extFactor: Float = -1.0f,
    attnFactor: Float = 1.0f,
    betaFast: Float = 32.0f,
    betaSlow: Float = 1.0f,
    origCtx: Int = 0
)

final case class BatchParams(size: Int = 512, threads: Int = Default.threads)

final case class ContextParams(
    seed: Int = -1,
    size: Int = 512,
    threads: Int = Default.threads,
    batch: BatchParams = BatchParams(),
    rope: RopeParams = RopeParams(),
    yarn: YarnParams = YarnParams()
)

final case class Penalty(
    repeat: Float = 1.10f,
    frequency: Float = .0f,
    presence: Float = .0f,
    penalizeNewLines: Boolean = true
)

final case class Dynatemp(
    range: Float = .0f,
    exponent: Float = 1.0f
)

enum Sampler:
  case TOP_K, TAIL_FREE, TYPICAL, TOP_P, MIN_P, TEMPERATURE

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
      topP: Float = .95f,
      minP: Float = .05f,
      dynatemp: Dynatemp = Dynatemp(),
      samplers: List[Sampler] = Sampler.values.toList
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
