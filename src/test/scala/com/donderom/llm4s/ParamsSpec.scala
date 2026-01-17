package com.donderom.llm4s

import org.scalatest.*
import flatspec.*
import matchers.*

class ParamsSpec extends AnyFlatSpec with should.Matchers:
  "ModelParams" should "validate number of GPU layers" in:
    val withGpuLayers = (layers: GpuLayers) =>
      ModelParams.parse(ModelParams(gpuLayers = layers))

    withGpuLayers(GpuLayers.Auto).isRight should be(true)
    withGpuLayers(GpuLayers.All).isRight should be(true)
    withGpuLayers(GpuLayers.None).isRight should be(true)
    withGpuLayers(GpuLayers(1)).isRight should be(true)

    withGpuLayers(GpuLayers(0)) should be (GpuLayers.error)
    withGpuLayers(GpuLayers(-1)) should be (GpuLayers.error)
    withGpuLayers(GpuLayers(-2)) should be (GpuLayers.error)

  "LlmParams" should "validate context size" in:
    val withCtxSize = (ctxSize: ContextSize) =>
      LlmParams.parse(LlmParams(context = ContextParams(size = ctxSize)))

    withCtxSize(ContextSize.Auto).isRight should be(true)
    withCtxSize(ContextSize(1)).isRight should be(true)

    withCtxSize(ContextSize(0)) should be (ContextSize.error)
    withCtxSize(ContextSize(-1)) should be (ContextSize.error)
