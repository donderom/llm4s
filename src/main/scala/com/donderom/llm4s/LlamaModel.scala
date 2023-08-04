package com.donderom.llm4s

import java.nio.file.Path

import scala.util.Try

import fr.hammons.slinc.runtime.given
import fr.hammons.slinc.{FSet, Ptr, Scope, Slinc}

extension (bool: Boolean) def toByte: Byte = if bool then 1 else 0

trait LlamaModel:
  case class Model(repr: Llama.Model, params: llama_context_params)

  val embedding: Boolean

  private lazy val binding = Try(FSet.instance[Llama])

  def createModel(model: Path, params: ContextParams): Try[Model] =
    binding.foreach(_.llama_backend_init(params.numa.toByte))

    val defaultParams = binding.map: llama =>
      llama.llama_context_default_params()

    Scope.confined:
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
      defaultParams: llama_context_params,
      params: ContextParams
  ): llama_context_params =
    defaultParams.copy(
      seed = params.seed,
      n_ctx = params.contextSize,
      n_batch = params.batchSize,
      n_gpu_layers = params.gpuLayers,
      main_gpu = params.mainGpu,
      low_vram = params.lowVram.toByte,
      f16_kv = params.f16.toByte,
      logits_all = 0,
      vocab_only = 0,
      use_mmap = params.mmap.toByte,
      use_mlock = params.mlock.toByte,
      embedding = embedding.toByte
    )
