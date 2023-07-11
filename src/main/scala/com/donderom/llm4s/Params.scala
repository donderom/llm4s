package com.donderom.llm4s

import java.nio.file.Path

case class ContextParams(
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
    lora: LoraParams = LoraParams()
)

case class LoraParams(
    adapter: Option[Path] = None,
    base: Option[Path] = None
)

object Mirostat:
  sealed trait Version
  case object V1 extends Version
  case object V2 extends Version

  case class Params(
      version: Version,
      tau: Float = 5.0f,
      eta: Float = .1f,
      am: Int = 100
  ):
    val mu: Float = 2.0f * tau

case class SamplingParams(
    temp: Float = .80f,
    repeatLastTokens: Int = 64,
    repeatPenalty: Float = 1.10f,
    frequencyPenalty: Float = .0f,
    presencePenalty: Float = .0f,
    mirostat: Option[Mirostat.Params] = None,
    topK: Option[Int] = Some(40),
    tfsZ: Float = 1.0f,
    typicalP: Float = 1.0f,
    topP: Float = .95f
)

case class LlmParams(
    context: ContextParams = ContextParams(),
    sampling: SamplingParams = SamplingParams(),
    predictTokens: Int = -1,
    keepTokens: Int = 0,
    suffix: Option[String] = None
)
