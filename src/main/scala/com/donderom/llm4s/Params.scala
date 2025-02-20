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
    // Target entropy
    val tau: Float = 5.0f
    // Learning rate
    val eta: Float = .1f

final case class AdapterParams(
    // Path to LoRA adapter GGUF file
    path: Path,
    // Custom scaling of the LoRA adapter
    scale: Float = 1.0f
)

object AdapterParams:
  def parse(params: AdapterParams): Result[AdapterParams] =
    if Files.exists(params.path) then Right(params)
    else Left(ConfigError(s"LoRA adapter file ${params.path} does not exist"))

final case class ModelParams(
    // Number of layers to store in VRAM
    gpuLayers: Int = -1,
    // GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
    mainGpu: Int = 0,
    // Use mmap if possible
    mmap: Boolean = true,
    // Force system to keep model in RAM
    mlock: Boolean = false,
    // Attempt optimizations on some NUMA systems
    numa: NumaStrategy = NumaStrategy.DISABLED
)

final case class RopeParams(
    scalingType: RopeScalingType = RopeScalingType.UNSPECIFIED,
    // RoPE base frequency, used by NTK-aware scaling
    freqBase: Float = 0.0f,
    // RoPE frequency scaling factor, expands context by a factor of 1/N
    freqScale: Float = 0.0f
)

final case class YarnParams(
    // Extrapolation mix factor
    extFactor: Float = -1.0f,
    // Magnitude scaling factor
    attnFactor: Float = 1.0f,
    // Low correction dim
    betaFast: Float = 32.0f,
    // High correction dim
    betaSlow: Float = 1.0f,
    // Original context size
    origCtx: Int = 0
)

final case class BatchParams(
    // Logical maximum batch size
    logical: Int = 2048,
    // Physical maximum batch size
    physical: Int = 512,
    // Number of threads to use for batch processing
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
    // Context size
    size: Int = 4096,
    // Number of threads to use for generation
    threads: Int = Default.threads,
    batch: BatchParams = BatchParams(),
    rope: RopeParams = RopeParams(),
    yarn: YarnParams = YarnParams(),
    // Whether to use flash attention
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
    // Last n tokens to penalize
    lastN: Int = 64,
    // Penalize repeat sequence of tokens
    repeat: Float = 1.0f,
    // Repeat alpha frequency penalty
    frequency: Float = .0f,
    // Repeat alpha presence penalty
    presence: Float = .0f
)

final case class Dry(
    // DRY repetition penalty for tokens extending repetition
    multiplier: Float = .0f,
    // multiplier * base ^ (length of sequence before token - allowed length)
    base: Float = 1.75f,
    // Tokens extending repetitions beyond this receive penalty
    allowedLength: Int = 2,
    // How many tokens to scan for repetitions
    penaltyLastN: Int = -1,
    // Sequence breakers
    seqBreakers: Seq[Char] = Seq[Char]('\n', ':', '"', '*')
)

final case class Xtc(
    probability: Float = .0f,
    threshold: Float = 0.10f
)

final case class Dynatemp(
    range: Float = .0f,
    // Controls how entropy maps to temperature in dynamic temperature sampler
    exponent: Float = 1.0f
)

enum SamplerType:
  case PENALTIES, DRY, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMPERATURE

enum Sampling:
  case Dist(
      // Whether to use greedy sampler
      greedy: Boolean = false,
      // List of samplers to apply (order is important)
      samplers: List[SamplerType] = SamplerType.values.toList,
      seed: Int = Default.seed,
      logitBias: Map[Int, Float] = Map(),
      penalty: Penalty = Penalty(),
      dry: Dry = Dry(),
      // Minimum number of tokens for samplers to to return
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
      // Target entropy
      tau: Float = Default.Mirostat.tau,
      // Learning rate
      eta: Float = Default.Mirostat.eta,
      // Maximum cross-entropy
      m: Int = 100
  )

  case Mirostat2(
      seed: Int = Default.seed,
      temp: Float = Default.temp,
      // Target entropy
      tau: Float = Default.Mirostat.tau,
      // Learning rate
      eta: Float = Default.Mirostat.eta
  )

enum Norm:
  case MaxAbsolute
  case Taxicab
  case Euclidean
  case PNorm(p: Int)

final case class EmbeddingParams(
    context: ContextParams = ContextParams(),
    // Normalisation for embeddings
    norm: Option[Norm] = None
)

object EmbeddingParams:
  def parse(params: EmbeddingParams): Result[EmbeddingParams] =
    for _ <- ContextParams.parse(params.context)
    yield params

final case class LlmParams(
    context: ContextParams = ContextParams(),
    sampling: Sampling = Sampling.Dist(),
    // Number of tokens to predict
    predictTokens: Option[Int] = None,
    // Number of tokens to keep from the initial prompt
    keepTokens: Int = 0,
    // Optional suffix appended to generated text
    suffix: Option[String] = None,
    // Whether to return prompt
    echo: Boolean = true,
    // List of stop sequences
    stopSeqs: List[String] = Nil,
    groupAttention: GroupAttention = GroupAttention(),
    // List of LoRA adapters
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
