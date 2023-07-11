package com.donderom.llm4s

import java.nio.file.Path

import scala.util.Try

import fr.hammons.slinc.runtime.given
import fr.hammons.slinc.{FSet, Ptr, Scope}

trait Llm(val modelPath: Path):
  def generate(prompt: String, params: LlmParams): Try[LazyList[String]]

  def close(): Unit

  def apply(prompt: String, params: LlmParams): Try[LazyList[String]] =
    generate(prompt, params)

object Llm:
  def apply(lib: Path, model: Path, params: ContextParams): Llm =
    val binding = Try:
      System.load(lib.toAbsolutePath.toString)
      FSet.instance[Llama]

    binding.foreach(_.llama_init_backend())

    val defaultParams = binding.map: llama =>
      llama.llama_context_default_params()

    val llm = Scope.confined:
      val baseModel = for
        llama <- binding
        defaultParams <- defaultParams
      yield llama.llama_load_model_from_file(
        path_model = Ptr.copy(model.toAbsolutePath.toString),
        params = llamaParams(defaultParams, params)
      )

      params.lora.adapter.fold(baseModel)(loraAdapter =>
        val err =
          for
            llama <- binding
            llm <- baseModel
          yield llama.llama_model_apply_lora_from_file(
            model = llm,
            path_lora = Ptr.copy(loraAdapter.toAbsolutePath.toString),
            path_base_model =
              Ptr.copy(params.lora.base.fold("")(_.toAbsolutePath.toString)),
            n_threads = params.threads
          )
        err.filter(_ == 0).flatMap(_ => baseModel)
      )

    new Llm(model) {
      def generate(prompt: String, params: LlmParams): Try[LazyList[String]] =
        Scope.confined:
          val ctx = for
            llama <- binding
            llm <- llm
            defaultParams <- defaultParams
          yield llama.llama_new_context_with_model(
            model = llm,
            params = llamaParams(defaultParams, params.context)
          )
          ctx.filter(_ != null).map(new SlincLlm(_).generate(prompt, params))

      def close(): Unit =
        for
          llama <- binding
          llm <- llm
        do llama.llama_free_model(llm)
    }

  private def llamaParams(
      defaultParams: llama_context_params,
      params: ContextParams
  ): llama_context_params =
    defaultParams.copy(
      seed = params.seed,
      n_ctx = params.contextSize,
      n_batch = params.batchSize,
      n_gpu_layers = params.gpuLayers,
      main_gpu = params.mainGpu,
      low_vram = if params.lowVram then 1 else 0,
      f16_kv = if params.f16 then 1 else 0,
      logits_all = 0,
      vocab_only = 0,
      use_mmap = if params.mmap then 1 else 0,
      use_mlock = if params.mlock then 1 else 0,
      embedding = 0
    )
