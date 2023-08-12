package com.donderom.llm4s

final case class State[Tok: Stringy](
    remaining: State.Remaining,
    evaluated: State.Evaluated,
    partialBytes: Array[Byte],
    probs: Vector[Option[Probability]],
    stop: Stop.State[Tok]
):
  import State.*

  def regular(evaluated: Evaluated, stop: Stop.State[Tok]): State[Tok] =
    this.copy(
      remaining = remaining.decr,
      evaluated = evaluated,
      partialBytes = Array.empty[Byte],
      probs = Vector.empty[Option[Probability]],
      stop = stop
    )

  def partial(
      evaluated: Evaluated,
      bytes: Array[Byte],
      prob: Option[Probability]
  ): State[Tok] =
    this.copy(
      remaining = remaining.decr,
      evaluated = evaluated,
      partialBytes = bytes,
      probs = probs :+ prob
    )

object State:
  opaque type Remaining = Int

  object Remaining:
    def apply(value: Int): Remaining = value

  extension (rem: Remaining)
    def decr: Remaining = rem - 1
    def none: Boolean = rem == 0

  opaque type Evaluated = Int

  object Evaluated:
    def apply(value: Int): Evaluated = value
    def none: Evaluated = apply(0)

  extension (eval: Evaluated)
    def toInt: Int = eval
    def incr: Evaluated = +1
    def +(num: Int): Evaluated = eval + num

  def apply[Tok: Stringy](
      predictTokens: Int,
      evaluated: Evaluated
  ): State[Tok] =
    State[Tok](
      remaining = Remaining(predictTokens),
      evaluated = evaluated,
      partialBytes = Array.empty[Byte],
      probs = Vector.empty[Option[Probability]],
      stop = Stop.State[Tok]()
    )
