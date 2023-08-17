package com.donderom.llm4s

import java.nio.file.Path

import scala.util.Try

import fr.hammons.slinc.runtime.given
import fr.hammons.slinc.{FSet, Ptr, Scope, Slinc}

trait LlamaModel:
  case class Model(repr: Llama.Model, params: Llama.ContextParams)

  val embedding: Boolean

  private lazy val binding = Try(FSet.instance[Llama])

  def createModel(model: Path, params: ContextParams): Try[Model] =
    binding.foreach(_.llama_backend_init(params.numa))

    val defaultParams = binding.map(_.llama_context_default_params())

    Scope.global:
      val baseModel = for
        llama <- binding
        defaultParams <- defaultParams
      yield Model(
        llama.llama_load_model_from_file(
          path_model = Ptr.copy(model.toAbsolutePath.toString),
          params = llamaParams(defaultParams, params)
        ),
        defaultParams
      )

      params.lora.adapter.fold(baseModel): loraAdapter =>
        val err =
          for
            llama <- binding
            llm <- baseModel
            loraBase = params.lora.base.fold(Slinc.getRuntime().Null): base =>
              Ptr.copy(base.toAbsolutePath.toString)
          yield llama.llama_model_apply_lora_from_file(
            model = llm.repr,
            path_lora = Ptr.copy(loraAdapter.toAbsolutePath.toString),
            path_base_model = loraBase,
            n_threads = params.threads
          )
        err.filter(_ == 0).flatMap(_ => baseModel)

  def createContext(
      llm: Try[Model],
      contextParams: ContextParams
  ): Try[Llama.Ctx] =
    for
      llama <- binding
      llm <- llm
      ctx = llama.llama_new_context_with_model(
        model = llm.repr,
        params = llamaParams(llm.params, contextParams)
      ) if ctx != Slinc.getRuntime().Null
    yield ctx

  def close(llm: Try[Model]): Unit =
    for
      llama <- binding
      llm <- llm
    do
      llama.llama_free_model(llm.repr)
      llama.llama_backend_free()

  private def llamaParams(
      defaultParams: Llama.ContextParams,
      params: ContextParams
  ): Llama.ContextParams =
    defaultParams.copy(
      seed = params.seed,
      n_ctx = params.contextSize,
      n_batch = params.batchSize,
      n_gpu_layers = params.gpuLayers,
      main_gpu = params.mainGpu,
      rope_freq_base = params.rope.freqBase,
      rope_freq_scale = params.rope.freqScale,
      low_vram = params.lowVram,
      f16_kv = params.f16,
      logits_all = false,
      vocab_only = false,
      use_mmap = params.mmap,
      use_mlock = params.mlock,
      embedding = embedding
    )
