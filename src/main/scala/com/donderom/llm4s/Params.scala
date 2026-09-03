package com.donderom.llm4s

import java.nio.file.{Files, Path}

import LlmError.ConfigError
import Llama.{FlashAttentionType, LoadMode, NumaStrategy, RopeScalingType}

object Default:
  lazy val threads = Runtime.getRuntime.availableProcessors

  val seed: Int = 0xfffffff
  val temp: Float = .8f

  object Mirostat:
    // Target entropy
    val tau: Float = 5.0f
    // Learning rate
    val eta: Float = .1f

/** Params to configure a LoRA adapter.
  *
  * @param path
  *   path to LoRA adapter GGUF file
  * @param scale
  *   custom scaling of the LoRA adapter. Defaults to 1.0.
  */
final case class AdapterParams(
    path: Path,
    scale: Float = 1.0f
)

trait Validation[A]:
  extension (s: String) def left: Result[A] = Left(ConfigError(s))

  def parse(params: A): Result[A]

object AdapterParams extends Validation[AdapterParams]:
  def parse(params: AdapterParams): Result[AdapterParams] =
    if Files.exists(params.path) then Right(params)
    else s"LoRA adapter file ${params.path} does not exist".left

/** Number of layers to store in VRAM.
  */
enum GpuLayers:
  case Auto, All, None
  case Custom(num: Int)

object GpuLayers extends Validation[GpuLayers]:
  val error = "Number of GPU layers should be positive".left

  def apply(num: Int): GpuLayers = GpuLayers.Custom(num)

  def parse(gpuLayers: GpuLayers): Result[GpuLayers] =
    gpuLayers match
      case Auto | All | None        => Right(gpuLayers)
      case Custom(size) if size > 0 => Right(gpuLayers)
      case Custom(_)                => error

/** Params controlling llama.cpp model.
  *
  * @param gpuLayers
  *   number of [[GpuLayers]] to store in VRAM. Defaults to auto.
  * @param mainGpu
  *   GPU that is used for the entire model when split_mode is
  *   LLAMA_SPLIT_MODE_NONE. Defaults to 0.
  * @param loadMode
  *   model [[Llama.LoadMode]]. Defaults to MMAP.
  * @param numa
  *   [[Llama.NumaStrategy]] optimization. Defaults to disabled.
  */
final case class ModelParams(
    gpuLayers: GpuLayers = GpuLayers.Auto,
    mainGpu: Int = 0,
    loadMode: LoadMode = LoadMode.MMAP,
    numa: NumaStrategy = NumaStrategy.DISABLED
)

object ModelParams:
  def parse(params: ModelParams): Result[ModelParams] =
    for _ <- GpuLayers.parse(params.gpuLayers)
    yield params

/** RoPE params.
  *
  * @param scalingType
  *   RoPE scaling type [[Llama.RopeScalingType]]. Defaults to unspecified.
  * @param freqBase
  *   RoPE base frequency used by NTK-aware scaling. Defaults to model derived
  *   value.
  * @param freqScale
  *   RoPE frequency scaling factor, expands context by a factor of 1/N.
  *   Defaults to model derived value.
  */
final case class RopeParams(
    scalingType: RopeScalingType = RopeScalingType.UNSPECIFIED,
    freqBase: Float = 0.0f,
    freqScale: Float = 0.0f
)

/** YaRN params.
  *
  * @param extFactor
  *   YaRN extrapolation mix factor. Defaults to model derived value.
  * @param attnFactor
  *   YaRN magnitude scaling factor. Defaults to -1.
  * @param betaFast
  *   YaRN low correction dim. Defaults to -1.
  * @param betaSlow
  *   YaRN high correction dim. Defaults to -1.
  * @param origCtx
  *   YaRN original context size. Defaults to 0.
  */
final case class YarnParams(
    extFactor: Float = -1.0f,
    attnFactor: Float = -1.0f,
    betaFast: Float = -1.0f,
    betaSlow: Float = -1.0f,
    origCtx: Int = 0
)

/** Batch params.
  *
  * @param logical
  *   maximum logical batch size. Defaults to 2048.
  * @param physical
  *   maximum physical batch size. Defaults to 512.
  * @param threads
  *   number of threads to use for batch processing. Defaults to available
  *   cores.
  */
final case class BatchParams(
    logical: Int = 2048,
    physical: Int = 512,
    threads: Int = Default.threads
)

object BatchParams extends Validation[BatchParams]:
  def parse(params: BatchParams): Result[BatchParams] =
    if params.logical < 1 then "Logical batch size should be positive".left
    else if params.physical < 1 then "Batch size should be positive".left
    else if params.threads < 1 then "Batch threads should be positive".left
    else Right(params)

/** Group attention params.
  *
  * @param factor
  *   group attention factor. Defaults to 1.
  * @param width
  *   group attention width. Defaults to 512.
  */
final case class GroupAttention(factor: Int = 1, width: Int = 512)

object GroupAttention extends Validation[GroupAttention]:
  def parse(params: GroupAttention): Result[GroupAttention] =
    if params.factor <= 0 then "Group attention factor should be positive".left
    else if params.width % params.factor != 0 then
      "Group attention width should be a multiple of factor".left
    else Right(params)

/** Flash attention type.
  */
enum FlashAttention:
  case Auto, On, Off

  private[llm4s] def asType: FlashAttentionType = this match
    case Auto => FlashAttentionType.AUTO
    case On   => FlashAttentionType.ENABLED
    case Off  => FlashAttentionType.DISABLED

/** Context size.
  */
enum ContextSize:
  case Auto
  case Custom(size: Int)

object ContextSize extends Validation[ContextSize]:
  val error = "Context size should be positive".left

  def apply(size: Int): ContextSize = ContextSize.Custom(size)

  def parse(contextSize: ContextSize): Result[ContextSize] =
    contextSize match
      case Auto                     => Right(contextSize)
      case Custom(size) if size > 0 => Right(contextSize)
      case Custom(_)                => error

/** Params controlling llama.cpp context
  *
  * @param size
  *   [[ContextSize]]. Defaults to auto.
  * @param threads
  *   number of threads to use for generation. Defaults to available cores.
  * @param batch
  *   batch related params [[BatchParams]]
  * @param rope
  *   RoPE related params [[RopeParams]]
  * @param yarn
  *   YaRN related params [[YarnParams]]
  * @param flashAttention
  *   controls the usage of [[FlashAttention]]. Defaults to auto.
  */
final case class ContextParams(
    size: ContextSize = ContextSize.Auto,
    threads: Int = Default.threads,
    batch: BatchParams = BatchParams(),
    rope: RopeParams = RopeParams(),
    yarn: YarnParams = YarnParams(),
    flashAttention: FlashAttention = FlashAttention.Auto
)

object ContextParams extends Validation[ContextParams]:
  def parse(params: ContextParams): Result[ContextParams] =
    for
      _ <- ContextSize.parse(params.size)
      _ <- Either.cond(
        params.threads > 0,
        params,
        ConfigError("Context threads should be positive")
      )
      _ <- BatchParams.parse(params.batch)
    yield params

/** Penalty sampling params.
  *
  * @param lastN
  *   last n tokens to penalize. Defaults to 64.
  * @param repeat
  *   penalize repeat sequence of tokens. Defaults to disabled.
  * @param frequency
  *   repeat alpha frequency penalty. Defaults to disabled.
  * @param presence
  *   repeat alpha presence penalty. Defaults to disabled.
  */
final case class Penalty(
    lastN: Option[Int] = Some(64),
    repeat: Option[Float] = None,
    frequency: Option[Float] = None,
    presence: Option[Float] = None
)

/** DRY sampling params.
  *
  * @param multiplier
  *   DRY repetition penalty for tokens extending repetition. Defaults to
  *   disabled.
  * @param base
  *   multiplier * base ^ (length of sequence before token - allowed length).
  *   Defaults to 1.75.
  * @param allowedLength
  *   tokens extending repetitions beyond this receive penalty. Defaults to 2.
  * @param penaltyLastN
  *   how many tokens to scan for repetitions. Defaults to 64.
  * @param seqBreakers
  *   sequence breakers. Defaults to `\n`, `:`, `"`, and `*`.
  */
final case class Dry(
    multiplier: Option[Float] = None,
    base: Option[Float] = Some(1.75f),
    allowedLength: Int = 2,
    penaltyLastN: Option[Int] = Some(64),
    seqBreakers: Seq[Char] = Seq[Char]('\n', ':', '"', '*')
)

/** XTC sampling params.
  *
  * @param probability
  *   XTC probability. Defaults to disabled.
  * @param threshold
  *   XTC threshold. Defaults to 0.10.
  */
final case class Xtc(
    probability: Option[Float] = None,
    threshold: Option[Float] = Some(0.10f)
)

/** Dynamic temperature sampling params.
  *
  * @param range
  *   dynatemp range. Defaults to disabled.
  * @param exponent
  *   controls how entropy maps to temperature in dynamic temperature sampler.
  *   Defaults to 1.
  */
final case class Dynatemp(
    range: Option[Float] = None,
    exponent: Float = 1.0f
)

/** Adaptive-p sampling params.
  *
  * @param target
  *   select tokens near this probability (valid range 0.0 to 1.0). Defaults to
  *   disabled.
  * @param decay
  *   EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 to
  *   0.99). Defaults to 0.9.
  */
final case class AdaptiveP(target: Option[Float] = None, decay: Float = 0.90f)

object AdaptiveP extends Validation[AdaptiveP]:
  val targetError = "Valid range for adaptive target is from 0.0 to 1.0".left
  val decayError =
    "Valid range for EMA decay for adaptation is from 0.0 to 0.99".left

  def parse(params: AdaptiveP): Result[AdaptiveP] =
    (params.target, params.decay) match
      case (Some(target), _) if !inRange(target, 0.0f, 1.0f) => targetError
      case (_, decay) if !inRange(decay, 0.0f, 0.99f)        => decayError
      case _                                                 => Right(params)

  private def inRange(num: Float, min: Float, max: Float): Boolean =
    num >= min && num <= max

/** Sampler type.
  */
enum SamplerType:
  case PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC,
    TEMPERATURE

/** Sampling algorithm.
  */
enum Sampling:
  /** Default sampling type.
    *
    * @param greedy
    *   whether to use greedy sampler. Defaults to false.
    * @param samplers
    *   list of samplers to apply (order is important). Defaults to
    *   [[SamplerType]]s: `PENALTIES` → `DRY` → `TOP_N_SIGMA` → `TOP_K` →
    *   `TYPICAL_P` → `TOP_P` → `MIN_P` → `XTC` → `TEMPERATURE`.
    * @param seed
    *   RNG seed
    * @param logitBias
    *   map of logit biases. Defaults to none.
    * @param penalty
    *   configuration [[Penalty]] configuration
    * @param dry
    *   [[Dry]] configuration
    * @param minKeep
    *   minimum number of tokens for samplers to return. Defaults to none.
    * @param topK
    *   Top-K sampling value. Defaults to 40.
    * @param typicalP
    *   Locally Typical sampling value. Defaults to none.
    * @param topP
    *   nucleus sampling value. Defaults to 0.95.
    * @param minP
    *   ninimum P sampling value. Defaults to 0.05.
    * @param topNSigma
    *   top n sigma value. Defaults to none.
    * @param xtc
    *   configuration [[Xtc]] configuration
    * @param temp
    *   sampling temperature. Defaults to 0.8.
    * @param dynatemp
    *   dynamic temperature [[Dynatemp]] configuration
    * @param adaptiveP
    *   adaptive-P [[AdaptiveP]] configuration. Defaults to none.
    */
  case Dist(
      greedy: Boolean = false,
      samplers: List[SamplerType] = SamplerType.values.toList,
      seed: Int = Default.seed,
      logitBias: Map[Int, Float] = Map(),
      penalty: Penalty = Penalty(),
      dry: Dry = Dry(),
      minKeep: Option[Short] = None,
      topK: Option[Int] = Some(40),
      typicalP: Option[Float] = None,
      topP: Option[Float] = Some(0.95f),
      minP: Option[Float] = Some(0.05f),
      topNSigma: Option[Float] = None,
      xtc: Xtc = Xtc(),
      temp: Float = Default.temp,
      dynatemp: Dynatemp = Dynatemp(),
      adaptiveP: Option[AdaptiveP] = None
  )

  /** Mirostat 1.0 sampling algorithm.
    *
    * @param seed
    *   RNG seed
    * @param temp
    *   sampling temperature
    * @param tau
    *   target entropy
    * @param eta
    *   learning rate
    * @param m
    *   maximum cross-entropy
    */
  case Mirostat1(
      seed: Int = Default.seed,
      temp: Float = Default.temp,
      tau: Float = Default.Mirostat.tau,
      eta: Float = Default.Mirostat.eta,
      m: Int = 100
  )

  /** Mirostat 2.0 sampling algorithm.
    *
    * @param seed
    *   RNG seed
    * @param temp
    *   sampling temperature
    * @param tau
    *   target entropy
    * @param eta
    *   learning rate
    */
  case Mirostat2(
      seed: Int = Default.seed,
      temp: Float = Default.temp,
      tau: Float = Default.Mirostat.tau,
      eta: Float = Default.Mirostat.eta
  )

object Sampling extends Validation[Sampling]:
  val minKeepError = "MinKeep should be positive".left
  val topKError = "Top-K should be positive".left
  val dryPenaltyLastNError = "Dry penalty last n cannot be negative".left
  val penaltyLastNError = "Penalty last n cannot be negative".left

  def parse(params: Sampling): Result[Sampling] =
    params match
      case dist: Sampling.Dist =>
        for
          _ <- AdaptiveP.parse(dist.adaptiveP.getOrElse(AdaptiveP()))
          params <- parseDist(dist)
        yield params
      case _: Mirostat1 | _: Mirostat2 => Right(params)

  private def parseDist(dist: Sampling.Dist): Result[Sampling] =
    if dist.minKeep.fold(false)(_ <= 0) then minKeepError
    else if dist.topK.fold(false)(_ <= 0) then topKError
    else if dist.dry.penaltyLastN.fold(false)(_ < 1) then dryPenaltyLastNError
    else if dist.penalty.lastN.fold(false)(_ < 1) then penaltyLastNError
    else Right(dist)

/** Embedding normalization type.
  */
enum Norm:
  case MaxAbsolute
  case Taxicab
  case Euclidean
  case PNorm(p: Int)

/** Params controlling embeddings generation.
  *
  * @param context
  *   [[ContextParams]]
  * @param poolingType
  *   [[Llama.PoolingType]] for embeddings
  * @param norm
  *   [[Norm]]alization type
  */
final case class EmbeddingParams(
    context: ContextParams = ContextParams(),
    poolingType: Llama.PoolingType = Llama.PoolingType.NONE,
    norm: Option[Norm] = None
)

object EmbeddingParams:
  def parse(params: EmbeddingParams): Result[EmbeddingParams] =
    for _ <- ContextParams.parse(params.context)
    yield params

/** An entry point for LLM configuration.
  *
  * @param context
  *   [[ContextParams]] such as context size, batch sizes, attention type, etc.
  * @param sampling
  *   params controlling [[Sampling]]
  * @param predictTokens
  *   number of tokens to predict. Defaults to unlimited.
  * @param keepTokens
  *   number of tokens to keep from the initial prompt. Defaults to 0.
  * @param suffix
  *   optional suffix appended to generated text. Defaults to none.
  * @param echo
  *   whether to return prompt. Defaults to true.
  * @param stopSeqs
  *   list of stop sequences. Defaults to none.
  * @param groupAttention
  *   [[GroupAttention]] configuration
  * @param lora
  *   list of LoRA adapters [[AdapterParams]]. Defaults to none.
  */
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
      _ <- Sampling.parse(params.sampling)
      _ <- GroupAttention.parse(params.groupAttention)
    yield params
