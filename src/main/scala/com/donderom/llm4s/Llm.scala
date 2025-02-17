package com.donderom.llm4s

import fr.hammons.slinc.runtime.given
import fr.hammons.slinc.types.SizeT
import fr.hammons.slinc.{FSet, Ptr, Scope, Slinc}

import java.nio.file.Path

import scala.util.{Success, Try}

final case class Logprob(token: String, value: Double)
final case class Probability(logprob: Logprob, candidates: Array[Logprob])
final case class Token(value: String, probs: Vector[Probability] = Vector.empty)
final case class Usage(promptSize: Int, tokens: LazyList[Token])

trait Llm(val modelPath: Path) extends AutoCloseable:
  def generate(prompt: String, params: LlmParams): Try[Usage]

  def embeddings(prompt: String): Try[Array[Float]] =
    embeddings(prompt, EmbeddingParams())

  def embeddings(prompt: String, params: EmbeddingParams): Try[Array[Float]]

  def apply(prompt: String): Try[LazyList[String]] = apply(prompt, LlmParams())

  def apply(prompt: String, params: LlmParams): Try[LazyList[String]] =
    generate(prompt, params).map(_.tokens.map(_.value))

object Llm:
  def apply(model: Path): Llm = apply(model, ModelParams())

  def apply(model: Path, params: ModelParams): Llm =
    new Llm(model):
      val binding = Try(FSet.instance[Llama])
      val llm = createModel(model, params)

      def generate(prompt: String, params: LlmParams): Try[Usage] =
        for
          llm <- llm
          ctx <- createContext(llm, params.context, false)
          _ <- loadLora(llm, ctx, params.lora)
        yield SlincLlm(ctx).generate(prompt, params)

      def embeddings(
          prompt: String,
          params: EmbeddingParams
      ): Try[Array[Float]] =
        for
          llm <- llm
          ctx <- createContext(llm, params.context, true)
        yield SlincLlm(ctx).embeddings(prompt, params)

      def close(): Unit =
        for
          llama <- binding
          llm <- llm
        do
          llama.llama_model_free(llm)
          llama.llama_backend_free()

      private def createModel(
          model: Path,
          params: ModelParams
      ): Try[Llama.Model] =
        binding.map: llama =>
          llama.llama_backend_init()
          llama.llama_numa_init(params.numa)
          Scope.confined:
            llama.llama_model_load_from_file(
              path_model = Ptr.copy(model.toAbsolutePath.toString),
              params = llama.llama_model_default_params().copy(
                n_gpu_layers = params.gpuLayers,
                main_gpu = params.mainGpu,
                use_mmap = params.mmap,
                use_mlock = params.mlock
              )
            )

      private def createContext(
          llm: Llama.Model,
          contextParams: ContextParams,
          embedding: Boolean
      ): Try[Llama.Ctx] =
        for
          llama <- binding
          ctx = llama.llama_init_from_model(
            model = llm,
            params = llamaParams(
              llama.llama_context_default_params(),
              contextParams,
              embedding
            )
          ) if ctx != Slinc.getRuntime().Null
        yield ctx

      private def loadLora(
          llm: Llama.Model,
          ctx: Llama.Ctx,
          lora: List[AdapterParams]
      ): Try[Unit] =
        lora.map(loadAdapter(llm, ctx, _)).foldLeft(Try(())):
          case (acc, Success(_)) => acc
          case (_, failure)      => failure

      private def loadAdapter(
          llm: Llama.Model,
          ctx: Llama.Ctx,
          params: AdapterParams
      ): Try[Unit] =
        Scope.confined:
          for
            llama <- binding
            adapter <- Try(
              llama.llama_adapter_lora_init(
                model = llm,
                path_lora = Ptr.copy(params.path.toAbsolutePath.toString)
              )
            )
            if adapter != Slinc.getRuntime().Null
            _ <- Try(
              llama.llama_set_adapter_lora(
                ctx = ctx,
                adapter = adapter,
                scale = params.scale
              )
            )
          yield ()

      private def llamaParams(
          defaultParams: Llama.ContextParams,
          params: ContextParams,
          embedding: Boolean
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
