package com.donderom.llm4s

import java.nio.file.Path

import Llama.{NumaStrategy, RopeScalingType}

object Default:
  val threads = Runtime.getRuntime.availableProcessors

  val logprobs: Int = 0
  val seed: Int = 0xfffffff
  val temp: Float = .8f
  object Mirostat:
    val tau: Float = 5.0f
    val eta: Float = .1f
    val muCoef: Float = 2.0f

final case class AdapterParams(
    path: Path,
    scale: Float = 1.0f
)

final case class ModelParams(
    gpuLayers: Int = -1,
    mainGpu: Int = 0,
    mmap: Boolean = true,
    mlock: Boolean = false,
    numa: NumaStrategy = NumaStrategy.DISABLED
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

final case class BatchParams(
    logical: Int = 2048,
    physical: Int = 512,
    threads: Int = Default.threads
)

final case class GroupAttention(factor: Int = 1, width: Int = 512)

final case class ContextParams(
    size: Int = 4096,
    threads: Int = Default.threads,
    batch: BatchParams = BatchParams(),
    rope: RopeParams = RopeParams(),
    yarn: YarnParams = YarnParams()
)

final case class Penalty(
    lastN: Int = 64,
    repeat: Float = 1.0f,
    frequency: Float = .0f,
    presence: Float = .0f
)

final case class Dry(
    multiplier: Float = .0f,
    base: Float = 1.75f,
    allowedLength: Int = 2,
    penaltyLastN: Int = -1,
    seqBreakers: Seq[Char] = Seq[Char]('\n', ':', '"', '*')
)

final case class Xtc(
    probability: Float = .0f,
    threshold: Float = 0.10f
)

final case class Dynatemp(
    range: Float = .0f,
    exponent: Float = 1.0f
)

enum SamplerType:
  case PENALTIES, DRY, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE

enum Sampling:
  case Dist(
      greedy: Boolean = false,
      samplers: List[SamplerType] = SamplerType.values.toList,
      seed: Int = Default.seed,
      logitBias: Map[Int, Float] = Map(),
      penalty: Penalty = Penalty(),
      dry: Dry = Dry(),
      minKeep: Short = 0,
      topK: Int = 40,
      typicalP: Float = 1.0f,
      topP: Float = 0.95f,
      minP: Float = 0.05f,
      xtc: Xtc = Xtc(),
      temp: Float = Default.temp,
      dynatemp: Dynatemp = Dynatemp()
  )

  case Mirostat1(
      seed: Int = Default.seed,
      temp: Float = Default.temp,
      tau: Float = Default.Mirostat.tau,
      eta: Float = Default.Mirostat.eta,
      m: Int = 100
  )

  case Mirostat2(
      seed: Int = Default.seed,
      temp: Float = Default.temp,
      tau: Float = Default.Mirostat.tau,
      eta: Float = Default.Mirostat.eta
  )

final case class LlmParams(
    context: ContextParams = ContextParams(),
    sampling: Sampling = Sampling.Dist(),
    predictTokens: Int = -1,
    keepTokens: Int = 0,
    suffix: Option[String] = None,
    echo: Boolean = true,
    stopSeqs: List[String] = Nil,
    groupAttention: GroupAttention = GroupAttention(),
    lora: List[AdapterParams] = Nil
)
