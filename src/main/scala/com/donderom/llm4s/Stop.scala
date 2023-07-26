package com.donderom.llm4s

object Stop:
  type SeqIndex = Int

  enum Match:
    case Full(start: Int, end: Int)
    case Partial(start: Int)

  final case class Token(token: String, usedBy: Set[SeqIndex])

  final case class State(
      tokens: Vector[Token] = Vector.empty,
      prefixes: Map[SeqIndex, String] = Map.empty
  ):
    def deferred(suffix: Option[String]): Vector[String] =
      suffix.fold(tokens.map(_.token))(tokens.map(_.token) :+ _)

  enum Action:
    case Cont(state: State)
    case Emit(chunk: String | Vector[String], state: State)
    case Stop(chunk: Vector[String])

  final case class Acc(stopSeqs: List[String]):
    lazy val mapping: Map[String, Int] = stopSeqs.zipWithIndex.toMap

    def step(token: String, state: State): Action =
      @annotation.tailrec
      def action(seqs: List[String], state: State, usedBy: Set[Int]): Action =
        seqs match
          case seq :: tail =>
            val seqIdx = mapping.getOrElse(seq, -1)
            val prefix = state.prefixes.get(seqIdx)
            matchToken(token = token, stopSeq = seq, prefix = prefix) match
              // Full match
              case Some(Match.Full(start, end)) =>
                val (kept, free) =
                  state.tokens.partition(_.usedBy.contains(seqIdx))
                if start == 0 then
                  // Calculate possible token split to release
                  val storedSeq = kept.map(_.token).mkString + token.take(end)
                  val split = storedSeq.take(storedSeq.size - seq.size)
                  if split.isEmpty then Action.Stop(free.map(_.token))
                  else Action.Stop(free.map(_.token) :+ split)
                else Action.Stop(free.map(_.token) :+ token.take(start))

              // Partial match
              case Some(Match.Partial(offset)) =>
                val prfx = token.substring(offset)
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
                state.copy(tokens = state.tokens :+ Stop.Token(token, usedBy))
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
