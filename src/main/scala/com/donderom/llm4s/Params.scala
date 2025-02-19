package com.donderom.llm4s

import java.nio.file.{Files, Path}

import LlmError.ConfigError
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

object AdapterParams:
  def parse(params: AdapterParams): Result[AdapterParams] =
    if Files.exists(params.path) then Right(params)
    else Left(ConfigError(s"LoRA adapter file ${params.path} does not exist"))

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

object BatchParams:
  def parse(params: BatchParams): Result[BatchParams] =
    if params.logical < 1 then
      Left(ConfigError("Logical batch size should be positive"))
    else if params.physical < 1 then
      Left(ConfigError("Batch size should be positive"))
    else if params.threads < 1 then
      Left(ConfigError("Batch threads should be positive"))
    else Right(params)

final case class GroupAttention(factor: Int = 1, width: Int = 512)

object GroupAttention:
  def parse(params: GroupAttention): Result[GroupAttention] =
    if params.factor <= 0 then
      Left(ConfigError("Group attention factor should be positive"))
    else if params.width % params.factor != 0 then
      Left(ConfigError("Group attention width should be a multiple of factor"))
    else Right(params)

final case class ContextParams(
    size: Int = 4096,
    threads: Int = Default.threads,
    batch: BatchParams = BatchParams(),
    rope: RopeParams = RopeParams(),
    yarn: YarnParams = YarnParams(),
    flashAttention: Boolean = false
)

object ContextParams:
  def parse(params: ContextParams): Result[ContextParams] =
    val config =
      if params.size < 0 then
        Left(ConfigError("Context size should be positive"))
      else if params.threads < 1 then
        Left(ConfigError("Context threads should be positive"))
      else Right(params)
    for
      _ <- BatchParams.parse(params.batch)
      config <- config
    yield config

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

enum Norm:
  case MaxAbsolute
  case Taxicab
  case Euclidean
  case PNorm(p: Int)

final case class EmbeddingParams(
    context: ContextParams = ContextParams(),
    norm: Option[Norm] = None
)

object EmbeddingParams:
  def parse(params: EmbeddingParams): Result[EmbeddingParams] =
    for _ <- ContextParams.parse(params.context)
    yield params

final case class LlmParams(
    context: ContextParams = ContextParams(),
    sampling: Sampling = Sampling.Dist(),
    predictTokens: Option[Int] = None,
    keepTokens: Int = 0,
    suffix: Option[String] = None,
    echo: Boolean = true,
    stopSeqs: List[String] = Nil,
    groupAttention: GroupAttention = GroupAttention(),
    lora: List[AdapterParams] = Nil
)

object LlmParams:
  def parse(params: LlmParams): Result[LlmParams] =
    for
      _ <- Either.cond(
        params.predictTokens.fold(true)(_ >= 0),
        params,
        ConfigError("Number of tokens to predict cannot be negative")
      )
      _ <- ContextParams.parse(params.context)
      _ <- GroupAttention.parse(params.groupAttention)
    yield params
