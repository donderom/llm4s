package com.donderom.llm4s

import fr.hammons.slinc.runtime.given
import fr.hammons.slinc.types.SizeT
import fr.hammons.slinc.{FSet, Ptr, Scope, Slinc}

import java.nio.file.{Files, Path}

final case class Logprob(token: String, value: Double)
final case class Probability(logprob: Logprob, candidates: Array[Logprob])
final case class Token(value: String, probs: Vector[Probability] = Vector.empty)
final case class Usage(promptSize: Int, tokens: LazyList[Token])

enum LlmError(message: String) extends Exception(message):
  case ModelError(message: String) extends LlmError(message)
  case ConfigError(message: String) extends LlmError(message)

import LlmError.ModelError

type Result[A] = Either[LlmError, A]
object Result:
  def unit: Result[Unit] = Right(())

trait Llm(val modelPath: Path) extends AutoCloseable:
  def generate(prompt: String, params: LlmParams): Result[Usage]

  def embeddings(prompt: String): Result[Array[Float]] =
    embeddings(prompt, EmbeddingParams())

  def embeddings(prompt: String, params: EmbeddingParams): Result[Array[Float]]

  def apply(prompt: String): Result[LazyList[String]] =
    apply(prompt, LlmParams())

  def apply(prompt: String, params: LlmParams): Result[LazyList[String]] =
    generate(prompt, params).map(_.tokens.map(_.value))

object Llm:
  def apply(model: Path): Llm = apply(model, ModelParams())

  def apply(model: Path, params: ModelParams): Llm =
    new Llm(model):
      val api = catchNonFatal(FSet.instance[Llama])("Cannot load libllama")
      val llm = createModel(model, params)

      def generate(prompt: String, params: LlmParams): Result[Usage] =
        for
          llm <- llm
          config <- LlmParams.parse(params)
          ctx <- createContext(llm, config.context, llamaParams(false))
          _ <- loadLora(llm, ctx, config.lora)
        yield SlincLlm(ctx).generate(prompt, config)

      def embeddings(
          prompt: String,
          params: EmbeddingParams
      ): Result[Array[Float]] =
        for
          _ <- Either.cond(
            params.poolingType != Llama.PoolingType.RANK,
            params,
            LlmError.ConfigError("Rank pooling type is not supported")
          )
          llm <- llm
          config <- EmbeddingParams.parse(params)
          ctx <- createContext(
            llm,
            config.context,
            embeddingParams(params.poolingType)
          )
        yield SlincLlm(ctx).embeddings(prompt, config)

      def close(): Unit =
        for
          llama <- api
          llm <- llm
        do
          llama.llama_model_free(llm)
          llama.llama_backend_free()

      private def createModel(
          model: Path,
          params: ModelParams
      ): Result[Llama.Model] =
        val error = s"Cannot load the model $model"
        for
          llama <- api
          path <- Either.cond(
            Files.exists(model),
            model,
            ModelError(s"Model file $model does not exist")
          )
          _ <- catchNonFatal(llama.llama_backend_init())(
            "Cannot load libllama backend"
          )
          _ <- catchNonFatal(llama.llama_numa_init(params.numa))(
            s"Cannot init Numa (${params.numa})"
          )
          m <- catchNonFatal(
            Scope.confined:
              llama.llama_model_load_from_file(
                path_model = Ptr.copy(path.toAbsolutePath.toString),
                params = llama.llama_model_default_params().copy(
                  n_gpu_layers = params.gpuLayers,
                  main_gpu = params.mainGpu,
                  use_mmap = params.mmap,
                  use_mlock = params.mlock
                )
              )
          )(error).filterOrElse(notNull, ModelError(error))
        yield m

      private def createContext(
          llm: Llama.Model,
          params: ContextParams,
          nativeParams: (
              Llama.ContextParams,
              ContextParams
          ) => Llama.ContextParams
      ): Result[Llama.Ctx] =
        val error = s"Cannot initialize model context ($params)"
        for
          llama <- api
          ctx <- catchNonFatal(
            llama.llama_init_from_model(
              model = llm,
              params =
                nativeParams(llama.llama_context_default_params(), params)
            )
          )(error).filterOrElse(notNull, ModelError(error))
        yield ctx

      private def loadLora(
          llm: Llama.Model,
          ctx: Llama.Ctx,
          lora: List[AdapterParams]
      ): Result[Unit] =
        lora.map(loadAdapter(llm, ctx, _)).foldLeft(Result.unit):
          case (acc, Right(_)) => acc
          case (_, failure)    => failure

      private def loadAdapter(
          llm: Llama.Model,
          ctx: Llama.Ctx,
          params: AdapterParams
      ): Result[Unit] =
        val error = s"Cannot initialize LoRA adapter ($params)"
        for
          llama <- api
          config <- AdapterParams.parse(params)
          adapter <- catchNonFatal(
            Scope.confined:
              llama.llama_adapter_lora_init(
                model = llm,
                path_lora = Ptr.copy(config.path.toAbsolutePath.toString)
              )
          )(error).filterOrElse(notNull, ModelError(error))
          _ <- catchNonFatal(
            llama.llama_set_adapter_lora(
              ctx = ctx,
              adapter = adapter,
              scale = config.scale
            )
          )(error)
        yield ()

      private def llamaParams(
          embedding: Boolean
      )(
          defaultParams: Llama.ContextParams,
          params: ContextParams
      ): Llama.ContextParams =
        defaultParams.copy(
          n_ctx = params.size,
          n_batch = params.batch.logical,
          n_ubatch = params.batch.physical,
          n_threads = params.threads,
          n_threads_batch = params.batch.threads,
          rope_scaling_type = params.rope.scalingType,
          rope_freq_base = params.rope.freqBase,
          rope_freq_scale = params.rope.freqScale,
          yarn_ext_factor = params.yarn.extFactor,
          yarn_attn_factor = params.yarn.attnFactor,
          yarn_beta_fast = params.yarn.betaFast,
          yarn_beta_slow = params.yarn.betaSlow,
          yarn_orig_ctx = params.yarn.origCtx,
          flash_attn = params.flashAttention,
          embeddings = embedding
        )

      private def embeddingParams(
          poolingType: Llama.PoolingType
      )(
          defaultParams: Llama.ContextParams,
          params: ContextParams
      ): Llama.ContextParams =
        llamaParams(true)(defaultParams, params)
          .copy(pooling_type = poolingType)

  private def catchNonFatal[A](f: => A)(reason: => String): Result[A] =
    try Right(f)
    catch
      case t if scala.util.control.NonFatal(t) =>
        Left(ModelError(s"$reason: ${t.getMessage}"))

  private def notNull(ptr: Ptr[Any]): Boolean = ptr != Slinc.getRuntime().Null
