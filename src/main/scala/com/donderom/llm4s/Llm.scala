package com.donderom.llm4s

import java.nio.file.Path

import scala.util.Try

import fr.hammons.slinc.runtime.given
import fr.hammons.slinc.{FSet, Ptr, Scope, Slinc}

final case class Logprob(token: String, value: Double)
final case class Probability(logprob: Logprob, candidates: Array[Logprob])
final case class Token(value: String, probs: Vector[Probability] = Vector.empty)
final case class Usage(promptSize: Int, tokens: LazyList[Token])

trait Llm(val modelPath: Path) extends AutoCloseable:
  def generate(prompt: String, params: LlmParams): Try[Usage]

  def embeddings(prompt: String): Try[Array[Float]] =
    embeddings(prompt, ContextParams())

  def embeddings(prompt: String, params: ContextParams): Try[Array[Float]]

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
        for ctx <- createContext(llm, params.context, false)
        yield SlincLlm(ctx).generate(prompt, params)

      def embeddings(prompt: String, params: ContextParams): Try[Array[Float]] =
        for ctx <- createContext(llm, params, true)
        yield SlincLlm(ctx).embeddings(prompt, params.batch)

      def close(): Unit =
        for
          llama <- binding
          llm <- llm
        do
          llama.llama_free_model(llm)
          llama.llama_backend_free()

      private def createModel(
          model: Path,
          params: ModelParams
      ): Try[Llama.Model] =
        binding.foreach: llama =>
          llama.llama_backend_init()
          llama.llama_numa_init(params.numa)

        Scope.global:
          val baseModel = binding.map: llama =>
            llama.llama_load_model_from_file(
              path_model = Ptr.copy(model.toAbsolutePath.toString),
              params = llama.llama_model_default_params().copy(
                n_gpu_layers = params.gpuLayers,
                main_gpu = params.mainGpu,
                use_mmap = params.mmap,
                use_mlock = params.mlock
              )
            )

          params.lora.adapter.fold(baseModel): loraAdapter =>
            val err =
              for
                llama <- binding
                llm <- baseModel
                loraBase = params.lora.base.fold(Slinc.getRuntime().Null):
                  base => Ptr.copy(base.toAbsolutePath.toString)
              yield llama.llama_model_apply_lora_from_file(
                model = llm,
                path_lora = Ptr.copy(loraAdapter.toAbsolutePath.toString),
                scale = params.lora.scale,
                path_base_model = loraBase,
                n_threads = params.lora.threads
              )
            err.filter(_ == 0).flatMap(_ => baseModel)

      private def createContext(
          llm: Try[Llama.Model],
          contextParams: ContextParams,
          embedding: Boolean
      ): Try[Llama.Ctx] =
        for
          llama <- binding
          llm <- llm
          ctx = llama.llama_new_context_with_model(
            model = llm,
            params = llamaParams(
              llama.llama_context_default_params(),
              contextParams,
              embedding
            )
          ) if ctx != Slinc.getRuntime().Null
        yield ctx

      private def llamaParams(
          defaultParams: Llama.ContextParams,
          params: ContextParams,
          embedding: Boolean
      ): Llama.ContextParams =
        defaultParams.copy(
          seed = params.seed,
          n_ctx = params.size,
          n_batch = params.batch.size,
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
          embedding = embedding
        )
