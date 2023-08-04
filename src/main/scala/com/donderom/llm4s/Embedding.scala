package com.donderom.llm4s

import java.nio.file.Path

import scala.util.Try

trait Embedding(val modelPath: Path) extends AutoCloseable:
  val embedding: Boolean = true

  def apply(prompt: String, params: ContextParams): Try[Array[Float]]

object Embedding:
  def apply(model: Path, params: ContextParams): Embedding =
    new Embedding(model) with LlamaModel:
      val llm = createModel(model, params)

      def apply(prompt: String, params: ContextParams): Try[Array[Float]] =
        val ctx = createContext(llm, params)
        ctx.map(SlincLlm(_).embeddings(prompt, params))

      def close(): Unit = close(llm)
