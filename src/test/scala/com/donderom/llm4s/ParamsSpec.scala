package com.donderom.llm4s

import org.scalatest.*
import flatspec.*
import matchers.*

class ParamsSpec extends AnyFlatSpec with should.Matchers:
  "LlmParams" should "validate context size" in:
    val withCtxSize = (ctxSize: ContextSize) =>
      LlmParams.parse(LlmParams(context = ContextParams(size = ctxSize)))

    withCtxSize(ContextSize.Auto).isRight should be(true)
    withCtxSize(ContextSize.Custom(1)).isRight should be(true)

    withCtxSize(ContextSize.Custom(0)) should be (ContextSize.error)
    withCtxSize(ContextSize.Custom(-1)) should be (ContextSize.error)
