package com.donderom.llm4s

import java.nio.file.Path

import scala.util.Try

trait Llm(val modelPath: Path) extends AutoCloseable:
  val embedding: Boolean = false

  def generate(prompt: String, params: LlmParams): Try[LazyList[String]]

  def apply(prompt: String, params: LlmParams): Try[LazyList[String]] =
    generate(prompt, params)

object Llm:
  def apply(model: Path, params: ContextParams): Llm =
    new Llm(model) with LlamaModel:
      val llm = createModel(model, params)

      def generate(prompt: String, params: LlmParams): Try[LazyList[String]] =
        val ctx = createContext(llm, params.context)
        ctx.map(SlincLlm(_).generate(prompt, params))

      def close(): Unit = close(llm)
