package com.donderom.llm4s

import org.scalatest.*
import flatspec.*
import matchers.*

class StopSpec extends AnyFlatSpec with should.Matchers:
  val defaultTokens =
    Array[String]("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
  val defaultStream = stream(defaultTokens, _)

  "Acc" should "ignore empty sequences" in:
    defaultStream(Nil) should be(defaultTokens)

  it should "ignore empty string sequences" in:
    defaultStream(List("")) should be(defaultTokens)

  it should "not stop if there are no matches" in:
    defaultStream(List("x", "y", "z")) should be(defaultTokens)

  it should "stop if seq matches beginning of text" in:
    val prefixes = defaultTokens.scanLeft("")(_ + _).tail
    val seqs = prefixes.map(p => defaultStream(List(p)))
    all(seqs) shouldBe empty

  it should "stop if seq matches end of text" in:
    val suffixes =
      defaultTokens.scanRight("")(_ + _).slice(1, defaultTokens.size)
    val expected =
      defaultTokens.scanLeft("")(_ + _).slice(1, defaultTokens.size)
    val matches = suffixes.zip(expected).map: (s, e) =>
      defaultStream(List(s)).mkString == e
    all(matches) should be(true)

  it should "release in flight tokens if end of stream is reached" in:
    defaultStream(List("hijk", "hijabc")) should be(defaultTokens)

  it should "release all tokens when there is no match anymore" in:
    defaultStream(List("cdefga")) should be(defaultTokens)

  it should "release non-overlapping tokens only when there is no match" in:
    defaultStream(List("abcdg", "cde")) should contain only ("a", "b")

  it should "stop at first overlapping sequence" in:
    defaultStream(List("abc", "bc")) shouldBe empty
    defaultStream(List("bc", "abc")) should contain only ("a")

  it should "handle repeating tokens" in:
    stream(
      Array("cast", "int", "to", "int"),
      List("inttointeger", "inttoint")
    ) should contain only ("cast")

  it should "release prefix if token is to be split" in:
    stream(Array("mono", "ch"), List("no")) should contain only ("mo")
    stream(Array("zero", "rope"), List("rope")) should contain only ("zero")
    stream(Array("zero", "roro"), List("roro")) should contain only ("ze")
    stream(Array("zero", "oops"), List("oo")) should contain only ("zer")
    stream(Array("buzz", "word"), List("zw")) should contain only ("buz")

  it should "handle unicode tokens" in:
    stream(Array(",", " ", "😍"), List("😍")) should contain only (",", " ")

  it should "not join prefix if it's not contiguous" in:
    stream(
      Array("then", " fine", "-t", "une"),
      List("ne-t")
    ) should contain only ("then fi")

  private def stream(
      tokens: Array[String],
      stopSeqs: List[String]
  ): LazyList[String] =
    val stop = Stop.Acc[Token](stopSeqs)
    import stop.*
    def gen(remaining: Int, state: Stop.State[Token]): LazyList[String] =
      if remaining != 0 then
        val token = tokens(tokens.size - remaining)
        stop.step(Token(token), state) match
          case Action.Cont(state) => gen(remaining - 1, state)
          case Action.Emit(chunk: Token, state) =>
            chunk.show #:: gen(remaining - 1, state)
          case Action.Emit(chunk: Vector[Token], state) =>
            LazyList.from(chunk.map(_.show)) #::: gen(remaining - 1, state)
          case Action.Stop(chunk) => LazyList.from(chunk.map(_.show))
      else LazyList.from(state.deferred(None).map(_.show))
    gen(tokens.size, state = Stop.State())
