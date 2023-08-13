package com.donderom.llm4s

import java.nio.file.Path

import scala.util.Try

final case class Logprob(token: String, value: Double)
final case class Probability(logprob: Logprob, candidates: Array[Logprob])
final case class Token(value: String, probs: Vector[Probability] = Vector.empty)
final case class Usage(promptSize: Int, tokens: LazyList[Token])

trait Llm(val modelPath: Path) extends AutoCloseable:
  val embedding: Boolean = false

  def generate(prompt: String, params: LlmParams): Try[Usage]

  def apply(prompt: String, params: LlmParams): Try[LazyList[String]] =
    generate(prompt, params).map(_.tokens.map(_.value))

object Llm:
  def apply(model: Path, params: ContextParams): Llm =
    new Llm(model) with LlamaModel:
      val llm = createModel(model, params)

      def generate(prompt: String, params: LlmParams): Try[Usage] =
        val ctx = createContext(llm, params.context)
        ctx.map(SlincLlm(_).generate(prompt, params))

      def close(): Unit = close(llm)
