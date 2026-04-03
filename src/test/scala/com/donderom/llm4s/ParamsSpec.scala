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

  "Sampling" should "validate Dist params" in:
    import Sampling.Dist
    val withDist = (dist: Dist) => Sampling.parse(dist)

    withDist(Dist()).isRight should be(true)

    withDist(Dist(minKeep = Some(-1))) should be (Sampling.minKeepError)
    withDist(Dist(topK = Some(-1))) should be (Sampling.topKError)

    val dry = Dry(penaltyLastN = Some(-2))
    withDist(Dist(dry = dry)) should be (Sampling.dryPenaltyLastNError)

    val penalty = Penalty(lastN = Some(-2))
    withDist(Dist(penalty = penalty)) should be (Sampling.penaltyLastNError)

  "Sampling" should "validate adaptiveP params"
    val withAdaptiveP = (adaptiveP: AdaptiveP) =>
      Sampling.parse(Sampling.Dist(adaptiveP = Some(adaptiveP)))

    val withAdaptiveTarget = (target: Float) =>
      withAdaptiveP(AdaptiveP(target = Some(target)))

    val withAdaptiveDecay = (decay: Float) =>
      withAdaptiveP(AdaptiveP(decay = decay))

    withAdaptiveP(AdaptiveP()).isRight should be(true)

    withAdaptiveTarget(.0f).isRight should be(true)
    withAdaptiveTarget(.5f).isRight should be(true)
    withAdaptiveTarget(1.0f).isRight should be(true)
    withAdaptiveTarget(-0.1f) should be (AdaptiveP.targetError)
    withAdaptiveTarget(1.1f) should be (AdaptiveP.targetError)

    withAdaptiveDecay(.0f).isRight should be(true)
    withAdaptiveDecay(.5f).isRight should be(true)
    withAdaptiveDecay(.99f).isRight should be(true)
    withAdaptiveDecay(-0.1f) should be (AdaptiveP.decayError)
    withAdaptiveDecay(1.0f) should be (AdaptiveP.decayError)
