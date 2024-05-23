package com.donderom.llm4s

import fr.hammons.slinc.types.*
import fr.hammons.slinc.{CUnion, FSet, Ptr, Struct, Transform}

object Llama:
  type Pos = CInt
  type Token = CInt
  type SeqId = CInt
  type Ctx = Ptr[Any]
  type Model = Ptr[Any]
  type Grammar = Ptr[Any]

  enum VocabType:
    case NONE, SPM, BPE, WPM

  given Transform[VocabType, CInt](VocabType.fromOrdinal, _.ordinal)

  enum TokenType:
    case UNDEFINED, NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE

  given Transform[TokenType, CInt](TokenType.fromOrdinal, _.ordinal)

  enum Ftype(val code: CInt):
    case ALL_F32 extends Ftype(0)
    case MOSTLY_F16 extends Ftype(1)
    case MOSTLY_Q4_0 extends Ftype(2)
    case MOSTLY_Q4_1 extends Ftype(3)
    case MOSTLY_Q4_1_SOME_F16 extends Ftype(4)
    case MOSTLY_Q8_0 extends Ftype(7)
    case MOSTLY_Q5_0 extends Ftype(8)
    case MOSTLY_Q5_1 extends Ftype(9)
    case MOSTLY_Q2_K extends Ftype(10)
    case MOSTLY_Q3_K_S extends Ftype(11)
    case MOSTLY_Q3_K_M extends Ftype(12)
    case MOSTLY_Q3_K_L extends Ftype(13)
    case MOSTLY_Q4_K_S extends Ftype(14)
    case MOSTLY_Q4_K_M extends Ftype(15)
    case MOSTLY_Q5_K_S extends Ftype(16)
    case MOSTLY_Q5_K_M extends Ftype(17)
    case MOSTLY_Q6_K extends Ftype(18)
    case MOSTLY_IQ2_XXS extends Ftype(19)
    case MOSTLY_IQ2_XS extends Ftype(20)
    case MOSTLY_Q2_K_S extends Ftype(21)
    case MOSTLY_IQ3_XS extends Ftype(22)
    case MOSTLY_IQ3_XXS extends Ftype(23)
    case MOSTLY_IQ1_S extends Ftype(24)
    case MOSTLY_IQ4_NL extends Ftype(25)
    case MOSTLY_IQ3_S extends Ftype(26)
    case MOSTLY_IQ3_M extends Ftype(27)
    case MOSTLY_IQ2_S extends Ftype(28)
    case MOSTLY_IQ2_M extends Ftype(29)
    case MOSTLY_IQ4_XS extends Ftype(30)
    case MOSTLY_IQ1_M extends Ftype(31)
    case MOSTLY_BF16 extends Ftype(32)
    case GUESSED extends Ftype(1024)

  given Transform[Ftype, CInt](Ftype.fromOrdinal, _.code)

  enum RopeScalingType(val code: CInt):
    case UNSPECIFIED extends RopeScalingType(-1)
    case NONE extends RopeScalingType(0)
    case LINEAR extends RopeScalingType(1)
    case YARN extends RopeScalingType(2)
    case MAX_VALUE extends RopeScalingType(2)

  given Transform[RopeScalingType, CInt](
    _ match
      case 0 => RopeScalingType.NONE
      case 1 => RopeScalingType.LINEAR
      case 2 => RopeScalingType.YARN
      case _ => RopeScalingType.UNSPECIFIED
    ,
    _.code
  )

  enum PoolingType(val code: CInt):
    case UNSPECIFIED extends PoolingType(-1)
    case NONE extends PoolingType(0)
    case MEAN extends PoolingType(1)
    case CLS extends PoolingType(2)

  given Transform[PoolingType, CInt](
    _ match
      case 0 => PoolingType.NONE
      case 1 => PoolingType.MEAN
      case 2 => PoolingType.CLS
      case _ => PoolingType.UNSPECIFIED
    ,
    _.code
  )

  enum SplitMode:
    case NONE, LAYER, ROW

  given Transform[SplitMode, CInt](SplitMode.fromOrdinal, _.ordinal)

  final case class TokenData(id: Token, logit: CFloat, p: CFloat) derives Struct

  final case class TokenDataArray(
      data: Ptr[TokenData],
      size: SizeT,
      sorted: CBool
  ) derives Struct

  final case class Batch(
      n_tokens: CInt,
      token: Ptr[Token],
      embd: Ptr[CFloat],
      pos: Ptr[Pos],
      n_seq_id: Ptr[CInt],
      seq_id: Ptr[Ptr[SeqId]],
      logits: Ptr[CInt],
      all_pos_0: Pos,
      all_pos_1: Pos,
      all_seq_id: SeqId
  ) derives Struct

  enum ModelKvOverrideType:
    case INT, FLOAT, BOOL

  given Transform[ModelKvOverrideType, CInt](
    ModelKvOverrideType.fromOrdinal,
    _.ordinal
  )

  final case class ModelKvOverride(
      tag: ModelKvOverrideType,
      key: CChar,
      value: CUnion[(CInt, CDouble, CBool, CChar)]
  ) derives Struct

  final case class ModelParams(
      n_gpu_layers: CInt,
      split_mode: SplitMode,
      main_gpu: CInt,
      tensor_split: Ptr[CFloat],
      progress_callback: Ptr[(CFloat, Unit) => Unit],
      progress_callback_user_data: Ptr[Unit],
      kv_overrides: Ptr[ModelKvOverride],
      vocab_only: CBool,
      use_mmap: CBool,
      use_mlock: CBool,
      check_tensors: CBool
  ) derives Struct

  enum GgmlType(val code: CInt):
    case F32 extends GgmlType(0)
    case F16 extends GgmlType(1)
    case Q4_0 extends GgmlType(2)
    case Q4_1 extends GgmlType(3)
    case Q5_0 extends GgmlType(6)
    case Q5_1 extends GgmlType(7)
    case Q8_0 extends GgmlType(8)
    case Q8_1 extends GgmlType(9)
    case Q2_K extends GgmlType(10)
    case Q3_K extends GgmlType(11)
    case Q4_K extends GgmlType(12)
    case Q5_K extends GgmlType(13)
    case Q6_K extends GgmlType(14)
    case Q8_K extends GgmlType(15)
    case IQ2_XXS extends GgmlType(16)
    case IQ2_XS extends GgmlType(17)
    case IQ3_XXS extends GgmlType(18)
    case IQ1_S extends GgmlType(19)
    case IQ4_NL extends GgmlType(20)
    case IQ3_S extends GgmlType(21)
    case IQ2_S extends GgmlType(22)
    case IQ4_XS extends GgmlType(23)
    case I8 extends GgmlType(24)
    case I16 extends GgmlType(25)
    case I32 extends GgmlType(26)
    case I64 extends GgmlType(27)
    case F64 extends GgmlType(28)
    case IQ1_M extends GgmlType(29)
    case BF16 extends GgmlType(30)
    case COUNT extends GgmlType(31)

  given Transform[GgmlType, CInt](
    _ match
      case 0  => GgmlType.F32
      case 1  => GgmlType.F16
      case 2  => GgmlType.Q4_0
      case 3  => GgmlType.Q4_1
      case 6  => GgmlType.Q5_0
      case 7  => GgmlType.Q5_1
      case 8  => GgmlType.Q8_0
      case 9  => GgmlType.Q8_1
      case 10 => GgmlType.Q2_K
      case 11 => GgmlType.Q3_K
      case 12 => GgmlType.Q4_K
      case 13 => GgmlType.Q5_K
      case 14 => GgmlType.Q6_K
      case 15 => GgmlType.Q8_K
      case 16 => GgmlType.IQ2_XXS
      case 17 => GgmlType.IQ2_XS
      case 18 => GgmlType.IQ3_XXS
      case 19 => GgmlType.IQ1_S
      case 20 => GgmlType.IQ4_NL
      case 21 => GgmlType.IQ3_S
      case 22 => GgmlType.IQ2_S
      case 23 => GgmlType.IQ4_XS
      case 24 => GgmlType.I8
      case 25 => GgmlType.I16
      case 26 => GgmlType.I32
      case 27 => GgmlType.I64
      case 28 => GgmlType.F64
      case 29 => GgmlType.IQ1_M
      case 30 => GgmlType.BF16
      case 31 => GgmlType.COUNT
    ,
    _.code
  )

  final case class ContextParams(
      seed: CInt,
      n_ctx: CInt,
      n_batch: CInt,
      n_ubatch: CInt,
      n_seq_max: CInt,
      n_threads: CInt,
      n_threads_batch: CInt,
      rope_scaling_type: RopeScalingType,
      pooling_type: PoolingType,
      rope_freq_base: CFloat,
      rope_freq_scale: CFloat,
      yarn_ext_factor: CFloat,
      yarn_attn_factor: CFloat,
      yarn_beta_fast: CFloat,
      yarn_beta_slow: CFloat,
      yarn_orig_ctx: CInt,
      defrag_thold: CFloat,
      cb_eval: Ptr[Any],
      cb_eval_user_data: Ptr[Any],
      type_k: GgmlType,
      type_v: GgmlType,
      logits_all: CBool,
      embeddings: CBool,
      offload_kqv: CBool,
      flash_attn: CBool,
      abort_callback: Ptr[Any],
      abort_callback_data: Ptr[Any]
  ) derives Struct

  final case class ModelQuantizeParams(
      nthread: CInt,
      ftype: Ftype,
      output_tensor_type: GgmlType,
      token_embedding_type: GgmlType,
      allow_requantize: CBool,
      quantize_output_tensor: CBool,
      only_copy: CBool,
      pure: CBool,
      keep_split: CBool,
      imatrix: Ptr[Any],
      kv_overrides: Ptr[Any]
  ) derives Struct

  enum NumaStrategy:
    case DISABLED, DISTRIBUTE, ISOLATE, NUMACTL, MIRROR, COUNT

  given Transform[NumaStrategy, CInt](NumaStrategy.fromOrdinal, _.ordinal)

  enum Gretype:
    case END, ALT, RULE_REF, CHAR, CHAR_NOT, CHAR_RNG_UPPER, CHAR_ALT

  given Transform[Gretype, CInt](Gretype.fromOrdinal, _.ordinal)

  final case class GrammarElement(gretype: Gretype, value: CInt) derives Struct

  final case class Timings(
      t_start_ms: CDouble,
      t_end_ms: CDouble,
      t_load_ms: CDouble,
      t_sample_ms: CDouble,
      t_p_eval_ms: CDouble,
      t_eval_ms: CDouble,
      n_sample: CInt,
      n_p_eval: CInt,
      n_eval: CInt
  ) derives Struct

  // Information associated with an individual cell in the KV cache view.
  final case class KvCacheViewCell(pos: Pos) derives Struct

  final case class KvCacheView(
      n_cells: CInt,
      n_seq_max: CInt,
      token_count: CInt,
      used_cells: CInt,
      max_contiguous: CInt,
      max_contiguous_idx: CInt,
      cells: Ptr[KvCacheViewCell],
      cells_sequences: Ptr[SeqId]
  ) derives Struct

  final case class BeamView(
      tokens: Ptr[Token],
      n_tokens: SizeT,
      p: CFloat,
      eob: CBool
  ) derives Struct

  final case class BeamsState(
      beam_views: Ptr[BeamView],
      n_beams: SizeT,
      common_prefix_length: SizeT,
      last_call: CBool
  )

trait Llama derives FSet:
  import Llama.*

  def llama_model_default_params(): ModelParams
  def llama_context_default_params(): ContextParams
  def llama_model_quantize_default_params(): ModelQuantizeParams

  def llama_backend_init(): Unit

  def llama_numa_init(strategy: NumaStrategy): Unit

  def llama_backend_free(): Unit

  def llama_load_model_from_file(
      path_model: Ptr[CChar],
      params: ModelParams
  ): Model

  def llama_free_model(model: Model): Unit

  def llama_new_context_with_model(model: Model, params: ContextParams): Ctx

  def llama_free(ctx: Ctx): Unit

  def llama_time_us(): CInt

  def llama_max_devices(): CInt

  def llama_supports_mmap(): CBool
  def llama_supports_mlock(): CBool
  def llama_supports_gpu_offload(): CBool

  def llama_get_model(ctx: Ctx): Model

  def llama_n_ctx(ctx: Ctx): CInt
  def llama_n_batch(ctx: Ctx): CInt
  def llama_n_ubatch(ctx: Ctx): CInt
  def llama_n_seq_max(ctx: Ctx): CInt

  def llama_pooling_type(ctx: Ctx): PoolingType

  def llama_vocab_type(model: Model): VocabType

  def llama_n_vocab(model: Model): CInt
  def llama_n_ctx_train(model: Model): CInt
  def llama_n_embd(model: Model): CInt
  def llama_n_layer(model: Model): CInt

  // Get the model's RoPE frequency scaling factor
  def llama_rope_freq_scale_train(model: Model): CFloat

  // Get metadata value as a string by key name
  def llama_model_meta_val_str(
      model: Model,
      key: Ptr[CChar],
      buf: Ptr[CChar],
      buf_size: SizeT
  ): CInt

  // Get the number of metadata key/value pairs
  def llama_model_meta_count(model: Model): CInt

  // Get metadata key name by index
  def llama_model_meta_key_by_index(
      model: Model,
      i: CInt,
      buf: Ptr[CChar],
      buf_size: SizeT
  ): CInt

  // Get metadata value as a string by index
  def llama_model_meta_val_str_by_index(
      model: Model,
      i: CInt,
      buf: Ptr[CChar],
      buf_size: SizeT
  ): CInt

  // Get a string describing the model type
  def llama_model_desc(model: Model, buf: Ptr[CChar], buf_size: SizeT): CInt

  // Returns the total size of all the tensors in the model in bytes
  def llama_model_size(model: Model): CInt

  // Returns the total number of parameters in the model
  def llama_model_n_params(model: Model): CInt

  def llama_model_quantize(
      fname_inp: Ptr[CChar],
      fname_out: Ptr[CChar],
      params: Ptr[ModelQuantizeParams]
  ): CInt

  // Apply a LoRA adapter to a loaded model
  def llama_model_apply_lora_from_file(
      model: Model,
      path_lora: Ptr[CChar],
      scale: CFloat,
      path_base_model: Ptr[CChar],
      n_threads: CInt
  ): CInt

  def llama_control_vector_apply(
      lctx: Ctx,
      data: Ptr[Float],
      len: SizeT,
      n_embd: CInt,
      il_start: CInt,
      il_end: CInt
  ): CInt

  // KV

  // Create an empty KV cache view. (use only for debugging purposes)
  def llama_kv_cache_view_init(ctx: Ctx, n_seq_max: CInt): KvCacheView

  // Free a KV cache view. (use only for debugging purposes)
  def llama_kv_cache_view_free(view: Ptr[KvCacheView]): Unit

  // Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)
  def llama_kv_cache_view_update(ctx: Ctx, view: Ptr[KvCacheView]): Unit

  // Returns the number of tokens in the KV cache (slow, use only for debug)
  // If a KV cell has multiple sequences assigned to it, it will be counted multiple times
  def llama_get_kv_cache_token_count(ctx: Ctx): CInt

  // Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
  def llama_get_kv_cache_used_cells(ctx: Ctx): CInt

  // Clear the KV cache
  def llama_kv_cache_clear(ctx: Ctx): Unit

  // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
  // seq_id < 0 : match any sequence
  // p0 < 0     : [0,  p1]
  // p1 < 0     : [p0, inf)
  def llama_kv_cache_seq_rm(ctx: Ctx, seq_id: SeqId, p0: Pos, p1: Pos): CBool

  // Copy all tokens that belong to the specified sequence to another sequence
  // Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
  // p0 < 0 : [0,  p1]
  // p1 < 0 : [p0, inf)
  def llama_kv_cache_seq_cp(
      ctx: Ctx,
      seq_id_src: SeqId,
      seq_id_dst: SeqId,
      p0: Pos,
      p1: Pos
  ): Unit

  // Removes all tokens that do not belong to the specified sequence
  def llama_kv_cache_seq_keep(ctx: Ctx, seq_id: SeqId): Unit

  // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
  // If the KV cache is RoPEd, the KV data is updated accordingly
  // p0 < 0 : [0,  p1]
  // p1 < 0 : [p0, inf)
  def llama_kv_cache_seq_add(
      ctx: Ctx,
      seq_id: SeqId,
      p0: Pos,
      p1: Pos,
      delta: Pos
  ): Unit

  // Integer division of the positions by factor of `d > 1`
  // If the KV cache is RoPEd, the KV data is updated accordingly
  // p0 < 0 : [0,  p1]
  // p1 < 0 : [p0, inf)
  def llama_kv_cache_seq_div(
      ctx: Ctx,
      seq_id: SeqId,
      p0: Pos,
      p1: Pos,
      d: CInt
  ): Unit

  // Returns the largest position present in the KV cache for the specified sequence
  def llama_kv_cache_seq_pos_max(ctx: Ctx, seq_id: SeqId): Pos

  // Defragment the KV cache
  // This will be applied:
  //   - lazily on next llama_decode()
  //   - explicitly with llama_kv_cache_update()
  def llama_kv_cache_defrag(ctx: Ctx): Unit

  // Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
  def llama_kv_cache_update(ctx: Ctx): Unit

  // Decoding

  // Return batch for single sequence of tokens starting at pos_0
  //
  // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
  def llama_batch_get_one(
      tokens: Ptr[Token],
      n_tokens: CInt,
      pos_0: Pos,
      seq_id: SeqId
  ): Batch

  // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
  // Each token can be assigned up to n_seq_max sequence ids
  // The batch has to be freed with llama_batch_free()
  // If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
  // Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
  // The rest of the llama_batch members are allocated with size n_tokens
  // All members are left uninitialized
  def llama_batch_init(n_tokens: CInt, embd: CInt, n_seq_max: CInt): Batch

  // Frees a batch of tokens allocated with llama_batch_init()
  def llama_batch_free(batch: Batch): Unit

  // Positive return values does not mean a fatal error, but rather a warning.
  //   0 - success
  //   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
  // < 0 - error
  def llama_decode(ctx: Ctx, batch: Batch): CInt

  // Set the number of threads used for decoding
  // n_threads is the number of threads used for generation (single token)
  // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
  def llama_set_n_threads(
      ctx: Ctx,
      n_threads: CInt,
      n_threads_batch: CInt
  ): Unit

  // Set whether to use causal attention or not
  // If set to true, the model will only attend to the past tokens
  def llama_set_causal_attn(ctx: Ctx, causal_attn: CBool): Unit

  // Set abort callback
  def llama_set_abort_callback(
      ctx: Ctx,
      abort_callback: Ptr[Any],
      abort_callback_data: Ptr[Any]
  ): Unit

  // Wait until all computations are finished
  // This is automatically done when using one of the functions below to obtain the computation results
  // and is not necessary to call it explicitly in most cases
  def llama_synchronize(ctx: Ctx): Unit

  // Token logits obtained from the last call to llama_decode()
  // The logits for which llama_batch.logits[i] != 0 are stored contiguously
  // in the order they have appeared in the batch.
  // Rows: number of tokens for which llama_batch.logits[i] != 0
  // Cols: n_vocab
  def llama_get_logits(ctx: Ctx): Ptr[Float]

  // Logits for the ith token. For positive indices, Equivalent to:
  // llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
  // Negative indicies can be used to access logits in reverse order, -1 is the last logit.
  // returns NULL for invalid ids.
  def llama_get_logits_ith(ctx: Ctx, i: CInt): Ptr[Float]

  // Get all output token embeddings.
  // when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
  // the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
  // in the order they have appeared in the batch.
  // shape: [n_outputs*n_embd]
  // Otherwise, returns NULL.
  def llama_get_embeddings(ctx: Ctx): Ptr[Float]

  // Get the embeddings for the ith token. For positive indices, Equivalent to:
  // llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
  // Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
  // shape: [n_embd] (1-dimensional)
  // returns NULL for invalid ids.
  def llama_get_embeddings_ith(ctx: Ctx, i: CInt): Ptr[Float]

  // Get the embeddings for a sequence id
  // Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
  // shape: [n_embd] (1-dimensional)
  def llama_get_embeddings_seq(ctx: Ctx, seq_id: SeqId): Ptr[Float]

  // Vocab

  def llama_token_get_text(model: Model, token: Token): Ptr[CChar]

  def llama_token_get_score(model: Model, token: Token): CFloat

  def llama_token_get_type(model: Model, token: Token): TokenType

  // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
  def llama_token_is_eog(model: Model, token: Token): CBool

  def llama_token_bos(model: Model): Token
  def llama_token_eos(model: Model): Token
  def llama_token_cls(model: Model): Token
  def llama_token_sep(model: Model): Token
  def llama_token_nl(model: Model): Token

  // Returns -1 if unknown, 1 for true or 0 for false.
  def llama_add_bos_token(model: Model): CInt

  // Returns -1 if unknown, 1 for true or 0 for false.
  def llama_add_eos_token(model: Model): CInt

  def llama_token_prefix(model: Model): Token
  def llama_token_middle(model: Model): Token
  def llama_token_suffix(model: Model): Token
  def llama_token_eot(model: Model): Token

  // Tokenization

  /// @details Convert the provided text into tokens.
  /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
  /// @return Returns the number of tokens on success, no more than n_tokens_max
  /// @return Returns a negative number on failure - the number of tokens that would have been returned
  /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
  ///                      as plaintext. Does not insert a leading space.
  def llama_tokenize(
      model: Model,
      text: Ptr[CChar],
      text_len: CInt,
      tokens: Ptr[Token],
      n_tokens_max: CInt,
      add_special: CBool,
      parse_special: CBool
  ): CInt

  // Token Id -> Piece.
  // Uses the vocabulary in the provided context.
  // Does not write null terminator to the buffer.
  // User code is responsible to remove the leading whitespace of the first non-BOS token when decoding multiple tokens.
  // @param special If true, special tokens are rendered in the output.
  def llama_token_to_piece(
      model: Model,
      token: Token,
      buf: Ptr[CChar],
      length: CInt,
      special: CBool
  ): CInt

  // Grammar

  def llama_grammar_init(
      rules: Ptr[Ptr[GrammarElement]],
      n_rules: SizeT,
      start_rule_index: SizeT
  ): Grammar

  def llama_grammar_free(grammar: Grammar): Unit

  def llama_grammar_copy(grammar: Grammar): Grammar

  // Sampling functions

  // Sets the current rng seed.
  def llama_set_rng_seed(ctx: Ctx, seed: CInt): Unit

  /// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
  /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
  def llama_sample_repetition_penalties(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      last_tokens: Ptr[Token],
      penalty_last_n: SizeT,
      penalty_repeat: CFloat,
      penalty_freq: CFloat,
      penalty_present: CFloat
  ): Unit

  /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
  def llama_sample_softmax(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray]
  ): Unit

  /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  def llama_sample_top_k(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      k: CInt,
      min_keep: SizeT
  ): Unit

  /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  def llama_sample_top_p(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      p: CFloat,
      min_keep: SizeT
  ): Unit

  /// @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
  def llama_sample_min_p(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      p: CFloat,
      min_keep: SizeT
  ): Unit

  /// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
  def llama_sample_tail_free(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      z: CFloat,
      min_keep: SizeT
  ): Unit

  /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
  def llama_sample_typical(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      p: CFloat,
      min_keep: SizeT
  ): Unit

  /// @details Dynamic temperature implementation described in the paper https://arxiv.org/abs/2309.02772.
  def llama_sample_entropy(
      ctx: Ctx,
      candidates_p: Ptr[TokenDataArray],
      min_temp: Float,
      max_temp: Float,
      exponent_val: Float
  ): Unit

  def llama_sample_temp(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      temp: CFloat
  ): Unit

  /// @details Apply constraints from grammar
  def llama_sample_grammar(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      grammar: Grammar
  ): Unit

  /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
  /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
  /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
  /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
  /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
  /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
  def llama_sample_token_mirostat(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      tau: CFloat,
      eta: CFloat,
      m: CInt,
      mu: Ptr[CFloat]
  ): Token

  /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
  /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
  /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
  /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
  /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
  def llama_sample_token_mirostat_v2(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      tau: CFloat,
      eta: CFloat,
      mu: Ptr[CFloat]
  ): Token

  /// @details Selects the token with the highest probability.
  ///          Does not compute the token probabilities. Use llama_sample_softmax() instead.
  def llama_sample_token_greedy(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray]
  ): Token

  /// @details Randomly selects a token from the candidates based on their probabilities.
  def llama_sample_token(ctx: Ctx, candidates: Ptr[TokenDataArray]): Token

  /// @details Accepts the sampled token into the grammar
  def llama_grammar_accept_token(
      ctx: Ctx,
      grammar: Grammar,
      token: Token
  ): Unit

  // Beam search

  def llama_beam_search(
      ctx: Ctx,
      callback: Ptr[(Ptr[Any], BeamsState) => Unit],
      callback_data: Ptr[Any],
      n_beams: SizeT,
      n_past: CInt,
      n_predict: CInt
  ): Unit

  // Performance and system information

  def llama_get_timings(ctx: Ctx): Timings

  def llama_print_timings(ctx: Ctx): Unit
  def llama_reset_timings(ctx: Ctx): Unit

  def llama_print_system_info(): Ptr[CChar]
