package com.donderom.llm4s

trait Stringy[A]:
  extension (a: A) def show: String
  extension (s: String) def token: A

given Stringy[Token] with
  extension (t: Token) def show: String = t.value
  extension (s: String) def token: Token = Token(s)

object Stop:
  type SeqIndex = Int

  enum Match:
    case Full(start: Int, end: Int)
    case Partial(start: Int)

  final case class TokenEntry[Tok](token: Tok, usedBy: Set[SeqIndex])

  final case class State[Tok: Stringy](
      tokens: Vector[TokenEntry[Tok]] = Vector.empty[TokenEntry[Tok]],
      prefixes: Map[SeqIndex, String] = Map.empty
  ):
    def deferred(suffix: Option[String]): Vector[Tok] =
      val buffer = tokens.map(_.token)
      suffix.fold(buffer)(buffer :+ _.token)

  trait Acc[Tok: Stringy](val stopSeqs: List[String]):
    enum Action:
      case Cont(state: State[Tok])
      case Emit(chunk: Tok | Vector[Tok], state: State[Tok])
      case Stop(chunk: Vector[Tok])

    lazy val mapping: Map[String, Int] = stopSeqs.zipWithIndex.toMap

    def step(token: Tok, state: State[Tok]): Action =
      @annotation.tailrec
      def action(
          seqs: List[String],
          state: State[Tok],
          usedBy: Set[Int]
      ): Action =
        seqs match
          case seq :: tail =>
            val seqIdx = mapping.getOrElse(seq, -1)
            val prefix = state.prefixes.get(seqIdx)
            val tokenText = token.show
            matchToken(token = tokenText, stopSeq = seq, prefix = prefix) match
              // Full match
              case Some(Match.Full(start, end)) =>
                val (kept, free) =
                  state.tokens.partition(_.usedBy.contains(seqIdx))
                val freeTokens = free.map(_.token)
                if start == 0 then
                  // Calculate possible token split to release
                  val stored = kept.map(_.token.show).mkString
                  val full = stored + tokenText.take(end)
                  val split = full.take(full.size - seq.size)
                  if split.isEmpty then Action.Stop(freeTokens)
                  else Action.Stop(freeTokens :+ split.token)
                else Action.Stop(freeTokens :+ tokenText.take(start).token)

              // Partial match
              case Some(Match.Partial(offset)) =>
                val prfx = tokenText.substring(offset)
                val newPrefix = prefix.fold(prfx)(_ + prfx)
                val prefixes = state.prefixes.updated(seqIdx, newPrefix)
                action(tail, state.copy(prefixes = prefixes), usedBy + seqIdx)

              // No match
              case None =>
                val newState = state.prefixes.get(seqIdx).fold(state): _ =>
                  // Release tokens from state if any
                  val tokens = state.tokens.collect:
                    case t if t.usedBy == Set(seqIdx) =>
                      t.copy(usedBy = Set())
                  val prefixes = state.prefixes.removed(seqIdx)
                  state.copy(tokens = tokens, prefixes = prefixes)
                action(tail, newState, usedBy)

          case Nil =>
            val newState =
              if !usedBy.isEmpty then
                state.copy(tokens = state.tokens :+ TokenEntry(token, usedBy))
              else state
            // State is empty, release token immediately
            if newState.tokens.isEmpty then Action.Emit(token, newState)
            else
              val (free, keep) = newState.tokens.partition(_.usedBy.isEmpty)
              if free.isEmpty then Action.Cont(newState)
              else if keep.isEmpty then
                Action.Emit(
                  free.map(_.token) :+ token,
                  newState.copy(tokens = keep)
                )
              else Action.Emit(free.map(_.token), newState.copy(tokens = keep))
      end action

      if stopSeqs.isEmpty then Action.Emit(token, state)
      else action(stopSeqs.filterNot(_.isEmpty).distinct, state, Set())
    end step

  object Acc:
    def apply[Token: Stringy](stopSeqs: List[String]): Acc[Token] =
      new Acc[Token](stopSeqs) {}

  private def matchToken(
      token: String,
      stopSeq: String,
      prefix: Option[String]
  ): Option[Match] =
    @annotation.tailrec
    def loop(start: Int, end: Int): Option[Match] =
      if end > token.size then
        val sub = token.substring(start, end - 1)
        if sub == stopSeq then Some(Match.Full(start, end - 1))
        else Option(sub).filterNot(_.isEmpty).map(_ => Match.Partial(start))
      else
        val sub = token.substring(start, end)
        if sub == stopSeq then Some(Match.Full(start, end))
        else if stopSeq.startsWith(sub) then loop(start, end + 1)
        else loop(start + 1, start + 2)

    prefix.fold(loop(0, 1)): p =>
      val contMatch = (1 to token.size)
        .takeWhile(i => stopSeq.startsWith(p + token.substring(0, i))).size
      val fullMatch = p.concat(token.take(contMatch)) == stopSeq
      if fullMatch then Some(Match.Full(0, contMatch))
      else if contMatch == token.size then Some(Match.Partial(0))
      else loop(0, 1)
