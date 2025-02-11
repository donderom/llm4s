package com.donderom.llm4s

import fr.hammons.slinc.types.*
import fr.hammons.slinc.{CUnion, FSet, Ptr, Struct, Transform}

object Llama:
  type Pos = CInt
  type Token = CInt
  type SeqId = CInt

  type Vocab = Ptr[Any]
  type Model = Ptr[Any]
  type Ctx = Ptr[Any]
  type Sampler = Ptr[Any]

  type LoraAdapter = Ptr[Any]

  enum VocabType:
    case NONE, SPM, BPE, WPM, UGM, RWKV

  given Transform[VocabType, CInt](VocabType.fromOrdinal, _.ordinal)

  enum RopeType(val code: CInt):
    case NONE extends RopeType(-1)
    case NORM extends RopeType(0)
    case NEOX extends RopeType(2)
    case MROPE extends RopeType(8)
    case VISION extends RopeType(24)

  given Transform[RopeType, CInt](
    _ match
      case RopeType.NONE.code   => RopeType.NONE
      case RopeType.NORM.code   => RopeType.NORM
      case RopeType.NEOX.code   => RopeType.NEOX
      case RopeType.MROPE.code  => RopeType.MROPE
      case RopeType.VISION.code => RopeType.VISION
    ,
    _.code
  )

  enum TokenAttr(val code: CInt):
    case UNDEFINED extends TokenAttr(0)
    case UNKNOWN extends TokenAttr(1 << 0)
    case UNUSED extends TokenAttr(1 << 1)
    case NORMAL extends TokenAttr(1 << 2)
    case CONTROL extends TokenAttr(1 << 3)
    case USER_DEFINED extends TokenAttr(1 << 4)
    case BYTE extends TokenAttr(1 << 5)
    case NORMALIZED extends TokenAttr(1 << 6)
    case LSTRIP extends TokenAttr(1 << 7)
    case RSTRIP extends TokenAttr(1 << 8)
    case SINGLE_WORD extends TokenAttr(1 << 9)

  given Transform[TokenAttr, CInt](
    _ match
      case TokenAttr.UNDEFINED.code    => TokenAttr.UNDEFINED
      case TokenAttr.UNKNOWN.code      => TokenAttr.UNKNOWN
      case TokenAttr.UNUSED.code       => TokenAttr.UNUSED
      case TokenAttr.NORMAL.code       => TokenAttr.NORMAL
      case TokenAttr.CONTROL.code      => TokenAttr.CONTROL
      case TokenAttr.USER_DEFINED.code => TokenAttr.USER_DEFINED
      case TokenAttr.BYTE.code         => TokenAttr.BYTE
      case TokenAttr.NORMALIZED.code   => TokenAttr.NORMALIZED
      case TokenAttr.LSTRIP.code       => TokenAttr.LSTRIP
      case TokenAttr.RSTRIP.code       => TokenAttr.RSTRIP
      case TokenAttr.SINGLE_WORD.code  => TokenAttr.SINGLE_WORD
    ,
    _.code
  )

  enum Ftype(val code: CInt):
    case ALL_F32 extends Ftype(0)
    case MOSTLY_F16 extends Ftype(1)
    case MOSTLY_Q4_0 extends Ftype(2)
    case MOSTLY_Q4_1 extends Ftype(3)
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
    case MOSTLY_TQ1_0 extends Ftype(36)
    case MOSTLY_TQ2_0 extends Ftype(37)
    case GUESSED extends Ftype(1024)

  given Transform[Ftype, CInt](
    _ match
      case Ftype.ALL_F32.code        => Ftype.ALL_F32
      case Ftype.MOSTLY_F16.code     => Ftype.MOSTLY_F16
      case Ftype.MOSTLY_Q4_0.code    => Ftype.MOSTLY_Q4_0
      case Ftype.MOSTLY_Q4_1.code    => Ftype.MOSTLY_Q4_1
      case Ftype.MOSTLY_Q8_0.code    => Ftype.MOSTLY_Q8_0
      case Ftype.MOSTLY_Q5_0.code    => Ftype.MOSTLY_Q5_0
      case Ftype.MOSTLY_Q5_1.code    => Ftype.MOSTLY_Q5_1
      case Ftype.MOSTLY_Q2_K.code    => Ftype.MOSTLY_Q2_K
      case Ftype.MOSTLY_Q3_K_S.code  => Ftype.MOSTLY_Q3_K_S
      case Ftype.MOSTLY_Q3_K_M.code  => Ftype.MOSTLY_Q3_K_M
      case Ftype.MOSTLY_Q3_K_L.code  => Ftype.MOSTLY_Q3_K_L
      case Ftype.MOSTLY_Q4_K_S.code  => Ftype.MOSTLY_Q3_K_S
      case Ftype.MOSTLY_Q4_K_M.code  => Ftype.MOSTLY_Q4_K_M
      case Ftype.MOSTLY_Q5_K_S.code  => Ftype.MOSTLY_Q5_K_S
      case Ftype.MOSTLY_Q5_K_M.code  => Ftype.MOSTLY_Q5_K_M
      case Ftype.MOSTLY_Q6_K.code    => Ftype.MOSTLY_Q6_K
      case Ftype.MOSTLY_IQ2_XXS.code => Ftype.MOSTLY_IQ2_XXS
      case Ftype.MOSTLY_IQ2_XS.code  => Ftype.MOSTLY_IQ2_XS
      case Ftype.MOSTLY_Q2_K_S.code  => Ftype.MOSTLY_Q2_K_S
      case Ftype.MOSTLY_IQ3_XS.code  => Ftype.MOSTLY_IQ3_XS
      case Ftype.MOSTLY_IQ3_XXS.code => Ftype.MOSTLY_IQ3_XXS
      case Ftype.MOSTLY_IQ1_S.code   => Ftype.MOSTLY_IQ1_S
      case Ftype.MOSTLY_IQ4_NL.code  => Ftype.MOSTLY_IQ4_NL
      case Ftype.MOSTLY_IQ3_S.code   => Ftype.MOSTLY_IQ3_S
      case Ftype.MOSTLY_IQ3_M.code   => Ftype.MOSTLY_IQ3_M
      case Ftype.MOSTLY_IQ2_S.code   => Ftype.MOSTLY_IQ2_S
      case Ftype.MOSTLY_IQ2_M.code   => Ftype.MOSTLY_IQ2_M
      case Ftype.MOSTLY_IQ4_XS.code  => Ftype.MOSTLY_IQ4_XS
      case Ftype.MOSTLY_IQ1_M.code   => Ftype.MOSTLY_IQ1_M
      case Ftype.MOSTLY_BF16.code    => Ftype.MOSTLY_BF16
      case Ftype.MOSTLY_TQ1_0.code   => Ftype.MOSTLY_TQ1_0
      case Ftype.MOSTLY_TQ2_0.code   => Ftype.MOSTLY_TQ2_0
      case Ftype.GUESSED.code        => Ftype.GUESSED
    ,
    _.code
  )

  enum RopeScalingType(val code: CInt):
    case UNSPECIFIED extends RopeScalingType(-1)
    case NONE extends RopeScalingType(0)
    case LINEAR extends RopeScalingType(1)
    case YARN extends RopeScalingType(2)
    case LONGROPE extends RopeScalingType(3)
    case MAX_VALUE extends RopeScalingType(3)

  given Transform[RopeScalingType, CInt](
    _ match
      case RopeScalingType.NONE.code     => RopeScalingType.NONE
      case RopeScalingType.LINEAR.code   => RopeScalingType.LINEAR
      case RopeScalingType.YARN.code     => RopeScalingType.YARN
      case RopeScalingType.LONGROPE.code => RopeScalingType.LONGROPE
      case _                             => RopeScalingType.UNSPECIFIED
    ,
    _.code
  )

  enum PoolingType(val code: CInt):
    case UNSPECIFIED extends PoolingType(-1)
    case NONE extends PoolingType(0)
    case MEAN extends PoolingType(1)
    case CLS extends PoolingType(2)
    case LAST extends PoolingType(3)
    case RANK extends PoolingType(4)

  given Transform[PoolingType, CInt](
    _ match
      case PoolingType.NONE.code => PoolingType.NONE
      case PoolingType.MEAN.code => PoolingType.MEAN
      case PoolingType.CLS.code  => PoolingType.CLS
      case PoolingType.LAST.code => PoolingType.LAST
      case PoolingType.RANK.code => PoolingType.RANK
      case _                     => PoolingType.UNSPECIFIED
    ,
    _.code
  )

  enum AttentionType(val code: CInt):
    case UNSPECIFIED extends AttentionType(-1)
    case CAUSAL extends AttentionType(0)
    case NON_CAUSAL extends AttentionType(1)

  given Transform[AttentionType, CInt](
    _ match
      case AttentionType.UNSPECIFIED.code => AttentionType.UNSPECIFIED
      case AttentionType.CAUSAL.code      => AttentionType.CAUSAL
      case AttentionType.NON_CAUSAL.code  => AttentionType.NON_CAUSAL
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
      selected: CInt,
      sorted: CBool
  ) derives Struct

  final case class Batch(
      n_tokens: CInt,
      token: Ptr[Token],
      embd: Ptr[CFloat],
      pos: Ptr[Pos],
      n_seq_id: Ptr[CInt],
      seq_id: Ptr[Ptr[SeqId]],
      logits: Ptr[CInt]
  ) derives Struct

  enum ModelKvOverrideType:
    case INT, FLOAT, BOOL, STR

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
      devices: Ptr[Any],
      n_gpu_layers: CInt,
      split_mode: SplitMode,
      main_gpu: CInt,
      tensor_split: Ptr[CFloat],
      // Callbacks are not supported yet
      progress_callback: Ptr[(CFloat, Any) => CBool],
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
    case TQ1_0 extends GgmlType(34)
    case TQ2_0 extends GgmlType(35)
    case COUNT extends GgmlType(39)

  given Transform[GgmlType, CInt](
    _ match
      case GgmlType.F32.code     => GgmlType.F32
      case GgmlType.F16.code     => GgmlType.F16
      case GgmlType.Q4_0.code    => GgmlType.Q4_0
      case GgmlType.Q4_1.code    => GgmlType.Q4_1
      case GgmlType.Q5_0.code    => GgmlType.Q5_0
      case GgmlType.Q5_1.code    => GgmlType.Q5_1
      case GgmlType.Q8_0.code    => GgmlType.Q8_0
      case GgmlType.Q8_1.code    => GgmlType.Q8_1
      case GgmlType.Q2_K.code    => GgmlType.Q2_K
      case GgmlType.Q3_K.code    => GgmlType.Q3_K
      case GgmlType.Q4_K.code    => GgmlType.Q4_K
      case GgmlType.Q5_K.code    => GgmlType.Q5_K
      case GgmlType.Q6_K.code    => GgmlType.Q6_K
      case GgmlType.Q8_K.code    => GgmlType.Q8_K
      case GgmlType.IQ2_XXS.code => GgmlType.IQ2_XXS
      case GgmlType.IQ2_XS.code  => GgmlType.IQ2_XS
      case GgmlType.IQ3_XXS.code => GgmlType.IQ3_XXS
      case GgmlType.IQ1_S.code   => GgmlType.IQ1_S
      case GgmlType.IQ4_NL.code  => GgmlType.IQ4_NL
      case GgmlType.IQ3_S.code   => GgmlType.IQ3_S
      case GgmlType.IQ2_S.code   => GgmlType.IQ2_S
      case GgmlType.IQ4_XS.code  => GgmlType.IQ4_XS
      case GgmlType.I8.code      => GgmlType.I8
      case GgmlType.I16.code     => GgmlType.I16
      case GgmlType.I32.code     => GgmlType.I32
      case GgmlType.I64.code     => GgmlType.I64
      case GgmlType.F64.code     => GgmlType.F64
      case GgmlType.IQ1_M.code   => GgmlType.IQ1_M
      case GgmlType.BF16.code    => GgmlType.BF16
      case GgmlType.TQ1_0.code   => GgmlType.TQ1_0
      case GgmlType.TQ2_0.code   => GgmlType.TQ2_0
      case GgmlType.COUNT.code   => GgmlType.COUNT
    ,
    _.code
  )

  final case class ContextParams(
      n_ctx: CInt,
      n_batch: CInt,
      n_ubatch: CInt,
      n_seq_max: CInt,
      n_threads: CInt,
      n_threads_batch: CInt,
      rope_scaling_type: RopeScalingType,
      pooling_type: PoolingType,
      attention_type: AttentionType,
      rope_freq_base: CFloat,
      rope_freq_scale: CFloat,
      yarn_ext_factor: CFloat,
      yarn_attn_factor: CFloat,
      yarn_beta_fast: CFloat,
      yarn_beta_slow: CFloat,
      yarn_orig_ctx: CInt,
      defrag_thold: CFloat,
      // Callbacks are not supported yet
      cb_eval: Ptr[Any],
      cb_eval_user_data: Ptr[Any],
      type_k: GgmlType,
      type_v: GgmlType,
      logits_all: CBool,
      embeddings: CBool,
      offload_kqv: CBool,
      flash_attn: CBool,
      no_perf: CBool,
      // Callbacks are not supported yet
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

  final case class LogitBias(token: Token, bias: CFloat) derives Struct

  final case class SamplerChainParams(no_perf: CBool) derives Struct

  final case class ChatMessage(role: Ptr[CChar], content: Ptr[CChar])
      derives Struct

trait Llama derives FSet:
  import Llama.*

  def llama_model_default_params(): ModelParams
  def llama_context_default_params(): ContextParams
  def llama_sampler_chain_default_params(): SamplerChainParams
  def llama_model_quantize_default_params(): ModelQuantizeParams

  // Initialize the llama + ggml backend
  // If numa is true, use NUMA optimizations
  // Call once at the start of the program
  def llama_backend_init(): Unit

  // Call once at the end of the program - currently only used for MPI
  def llama_backend_free(): Unit

  def llama_numa_init(strategy: NumaStrategy): Unit

  // Load the model from a file
  // If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
  // If the split file name does not follow this pattern, use llama_model_load_from_splits
  def llama_model_load_from_file(
      path_model: Ptr[CChar],
      params: ModelParams
  ): Model

  // Load the model from multiple splits (support custom naming scheme)
  // The paths must be in the correct order
  def llama_model_load_from_splits(
      paths: Ptr[Ptr[CChar]],
      n_paths: SizeT,
      params: ModelParams
  ): Model

  def llama_model_free(model: Model): Unit

  def llama_init_from_model(model: Model, params: ContextParams): Ctx

  // Frees all allocated memory
  def llama_free(ctx: Ctx): Unit

  def llama_time_us(): CInt

  def llama_max_devices(): SizeT

  def llama_supports_mmap(): CBool
  def llama_supports_mlock(): CBool
  def llama_supports_gpu_offload(): CBool
  def llama_supports_rpc(): CBool

  def llama_n_ctx(ctx: Ctx): CInt
  def llama_n_batch(ctx: Ctx): CInt
  def llama_n_ubatch(ctx: Ctx): CInt
  def llama_n_seq_max(ctx: Ctx): CInt

  def llama_get_model(ctx: Ctx): Model
  def llama_pooling_type(ctx: Ctx): PoolingType

  def llama_model_get_vocab(model: Model): Vocab
  def llama_model_rope_type(model: Model): RopeType

  def llama_model_n_ctx_train(model: Model): CInt
  def llama_model_n_embd(model: Model): CInt
  def llama_model_n_layer(model: Model): CInt
  def llama_model_n_head(model: Model): CInt

  // Get the model's RoPE frequency scaling factor
  def llama_model_rope_freq_scale_train(model: Model): CFloat

  def llama_vocab_type(vocab: Vocab): VocabType

  def llama_vocab_n_tokens(vocab: Vocab): CInt

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

  // Get the default chat template. Returns nullptr if not available
  // If name is NULL, returns the default chat template
  def llama_model_chat_template(model: Model, name: Ptr[CChar]): Ptr[CChar]

  // Returns the total number of parameters in the model
  def llama_model_n_params(model: Model): CInt

  // Returns true if the model contains an encoder that requires llama_encode() call
  def llama_model_has_encoder(model: Model): CBool

  // Returns true if the model contains a decoder that requires llama_decode() call
  def llama_model_has_decoder(model: Model): CBool

  // For encoder-decoder models, this function returns id of the token that must be provided
  // to the decoder to start generating output sequence. For other models, it returns -1.
  def llama_model_decoder_start_token(model: Model): Token

  // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
  def llama_model_is_recurrent(model: Model): CBool

  // Returns 0 on success
  def llama_model_quantize(
      fname_inp: Ptr[CChar],
      fname_out: Ptr[CChar],
      params: Ptr[ModelQuantizeParams]
  ): CInt

  // Adapters

  // Load a LoRA adapter from file
  def llama_adapter_lora_init(model: Model, path_lora: Ptr[CChar]): LoraAdapter

  // Manually free a LoRA adapter
  // Note: loaded adapters will be free when the associated model is deleted
  def llama_adapter_lora_free(adapter: LoraAdapter): Unit

  // Add a loaded LoRA adapter to given context
  // This will not modify model's weight
  def llama_set_adapter_lora(ctx: Ctx, adapter: LoraAdapter, scale: Float): CInt

  // Remove a specific LoRA adapter from given context
  // Return -1 if the adapter is not present in the context
  def llama_rm_adapter_lora(ctx: Ctx, adapter: LoraAdapter): CInt

  // Remove all LoRA adapters from given context
  def llama_clear_adapter_lora(ctx: Ctx): Unit

  // Apply a loaded control vector to a llama_context, or if data is NULL, clear
  // the currently loaded vector.
  // n_embd should be the size of a single layer's control, and data should point
  // to an n_embd x n_layers buffer starting from layer 1.
  // il_start and il_end are the layer range the vector should apply to (both inclusive)
  // See llama_control_vector_load in common to load a control vector.
  def llama_apply_adapter_cvec(
      ctx: Ctx,
      data: Ptr[Float],
      len: SizeT,
      n_embd: CInt,
      il_start: CInt,
      il_end: CInt
  ): CInt

  // KV cache

  // Returns the number of tokens in the KV cache (slow, use only for debug)
  // If a KV cell has multiple sequences assigned to it, it will be counted multiple times
  def llama_get_kv_cache_token_count(ctx: Ctx): CInt

  // Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
  def llama_get_kv_cache_used_cells(ctx: Ctx): CInt

  // Clear the KV cache - both cell info is erased and KV data is zeroed
  def llama_kv_cache_clear(ctx: Ctx): Unit

  // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
  // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
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
  // If the KV cache is RoPEd, the KV data is updated accordingly:
  //   - lazily on next llama_decode()
  //   - explicitly with llama_kv_cache_update()
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
  // If the KV cache is RoPEd, the KV data is updated accordingly:
  //   - lazily on next llama_decode()
  //   - explicitly with llama_kv_cache_update()
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

  // Check if the context supports KV cache shifting
  def llama_kv_cache_can_shift(ctx: Ctx): CBool

  // Decoding

  // Return batch for single sequence of tokens
  // The sequence ID will be fixed to 0
  // The position of the tokens will be tracked automatically by llama_decode
  //
  // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
  def llama_batch_get_one(tokens: Ptr[Token], n_tokens: CInt): Batch

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

  // Processes a batch of tokens with the ecoder part of the encoder-decoder model.
  // Stores the encoder output internally for later use by the decoder cross-attention layers.
  //   0 - success
  // < 0 - error. the KV cache state is restored to the state before this call
  def llama_encode(ctx: Ctx, batch: Batch): CInt

  // Positive return values does not mean a fatal error, but rather a warning.
  //   0 - success
  //   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
  // < 0 - error. the KV cache state is restored to the state before this call
  def llama_decode(ctx: Ctx, batch: Batch): CInt

  // Set the number of threads used for decoding
  // n_threads is the number of threads used for generation (single token)
  // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
  def llama_set_n_threads(
      ctx: Ctx,
      n_threads: CInt,
      n_threads_batch: CInt
  ): Unit

  // Get the number of threads used for generation of a single token.
  def llama_n_threads(ctx: Ctx): CInt

  // Get the number of threads used for prompt and batch processing (multiple token).
  def llama_n_threads_batch(ctx: Ctx): CInt

  // Set whether the model is in embeddings mode or not
  // If true, embeddings will be returned but logits will not
  def llama_set_embeddings(ctx: Ctx, embeddings: CBool): Unit

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
  // when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[1] with the rank of the sequence
  // otherwise: float[n_embd] (1-dimensional)
  def llama_get_embeddings_seq(ctx: Ctx, seq_id: SeqId): Ptr[Float]

  // Vocab

  def llama_vocab_get_text(vocab: Vocab, token: Token): Ptr[CChar]

  def llama_vocab_get_score(vocab: Vocab, token: Token): CFloat

  def llama_vocab_get_attr(vocab: Vocab, token: Token): TokenAttr

  // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
  def llama_vocab_is_eog(vocab: Vocab, token: Token): CBool

  // Identify if Token Id is a control token or a render-able token
  def llama_vocab_is_control(vocab: Vocab, token: Token): CBool

  // Special tokens
  def llama_vocab_bos(vocab: Vocab): Token
  def llama_vocab_eos(vocab: Vocab): Token
  def llama_vocab_eot(vocab: Vocab): Token
  def llama_vocab_sep(vocab: Vocab): Token
  def llama_vocab_nl(vocab: Vocab): Token
  def llama_vocab_pad(vocab: Vocab): Token

  def llama_vocab_get_add_bos(vocab: Vocab): CBool
  def llama_vocab_get_add_eos(vocab: Vocab): CBool

  def llama_vocab_fim_pre(vocab: Vocab): Token
  def llama_vocab_fim_suf(vocab: Vocab): Token
  def llama_vocab_fim_mid(vocab: Vocab): Token
  def llama_vocab_fim_pad(vocab: Vocab): Token
  def llama_vocab_fim_rep(vocab: Vocab): Token
  def llama_vocab_fim_sep(vocab: Vocab): Token

  // Tokenization

  /// @details Convert the provided text into tokens.
  /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
  /// @return Returns the number of tokens on success, no more than n_tokens_max
  /// @return Returns a negative number on failure - the number of tokens that would have been returned
  /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
  /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
  ///                      as plaintext. Does not insert a leading space.
  def llama_tokenize(
      vocab: Vocab,
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
  // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
  // @param special If true, special tokens are rendered in the output.
  def llama_token_to_piece(
      vocab: Vocab,
      token: Token,
      buf: Ptr[CChar],
      length: CInt,
      lstrip: CInt,
      special: CBool
  ): CInt

  /// @details Convert the provided tokens into text (inverse of llama_tokenize()).
  /// @param text The char pointer must be large enough to hold the resulting text.
  /// @return Returns the number of chars/bytes on success, no more than text_len_max.
  /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
  /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
  /// @param unparse_special If true, special tokens are rendered in the output.
  def llama_detokenize(
      vocab: Vocab,
      tokens: Ptr[Token],
      n_tokens: CInt,
      text: Ptr[CChar],
      text_len_max: CInt,
      remove_special: CBool,
      unparse_special: CBool
  ): CInt

  // Chat templates

  /// Apply chat template. Inspired by hf apply_chat_template() on python.
  /// Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
  /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
  /// @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
  /// @param chat Pointer to a list of multiple llama_chat_message
  /// @param n_msg Number of llama_chat_message in this chat
  /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
  /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
  /// @param length The size of the allocated buffer
  /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
  def llama_chat_apply_template(
      tmpl: Ptr[CChar],
      chat: Ptr[ChatMessage],
      n_msg: SizeT,
      add_ass: CBool,
      buf: Ptr[CChar],
      length: CInt
  ): CInt

  // Get list of built-in chat templates
  def llama_chat_builtin_templates(output: Ptr[Ptr[CChar]], len: SizeT): CInt

  // Sampling API

  def llama_sampler_name(sampler: Sampler): Ptr[CChar]
  def llama_sampler_accept(sampler: Sampler, token: Token): Unit
  def llama_sampler_apply(
      sampler: Sampler,
      candidates: Ptr[TokenDataArray]
  ): Unit
  def llama_sampler_reset(sampler: Sampler): Unit
  def llama_sampler_clone(sampler: Sampler): Sampler
  // important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
  def llama_sampler_free(sampler: Sampler): Unit

  def llama_sampler_chain_init(params: SamplerChainParams): Sampler

  // important: takes ownership of the sampler object and will free it when llama_sampler_free is called
  def llama_sampler_chain_add(chain: Sampler, smpl: Sampler): Unit
  def llama_sampler_chain_get(chain: Sampler, i: CInt): Sampler
  def llama_sampler_chain_n(chain: Sampler): CInt

  // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
  def llama_sampler_chain_remove(chain: Sampler, i: CInt): Sampler

  // Available samplers:

  def llama_sampler_init_greedy(): Sampler
  def llama_sampler_init_dist(seed: CInt): Sampler

  /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  def llama_sampler_init_top_k(k: CInt): Sampler

  /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
  def llama_sampler_init_top_p(p: CFloat, min_keep: SizeT): Sampler

  /// @details Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
  def llama_sampler_init_min_p(p: CFloat, min_keep: SizeT): Sampler

  /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
  def llama_sampler_init_typical(p: CFloat, min_keep: SizeT): Sampler

  /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
  def llama_sampler_init_temp(t: CFloat): Sampler

  /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
  def llama_sampler_init_temp_ext(
      t: CFloat,
      delta: CFloat,
      exponent: CFloat
  ): Sampler

  /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
  def llama_sampler_init_xtc(
      p: CFloat,
      t: CFloat,
      min_keep: SizeT,
      seed: CInt
  ): Sampler

  /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
  /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
  /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
  /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
  /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
  /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
  def llama_sampler_init_mirostat(
      n_vocab: CInt,
      seed: CInt,
      tau: CFloat,
      eta: CFloat,
      m: CInt
  ): Sampler

  /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
  /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
  /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
  /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
  /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
  def llama_sampler_init_mirostat_v2(
      seed: CInt,
      tau: CFloat,
      eta: CFloat
  ): Sampler

  def llama_sampler_init_grammar(
      vocab: Vocab,
      grammar_str: Ptr[CChar],
      grammar_root: Ptr[CChar]
  ): Sampler

  /// @details Lazy grammar sampler, introduced in https://github.com/ggerganov/llama.cpp/pull/9639
  /// @param trigger_words A list of words that will trigger the grammar sampler. This may be updated to a loose regex syntax (w/ ^) in a near future.
  /// @param trigger_tokens A list of tokens that will trigger the grammar sampler.
  def llama_sampler_init_grammar_lazy(
      vocab: Vocab,
      grammar_str: Ptr[CChar],
      grammar_root: Ptr[CChar],
      trigger_words: Ptr[Ptr[CChar]],
      num_trigger_words: SizeT,
      trigger_tokens: Ptr[Token],
      num_trigger_tokens: SizeT
  ): Sampler

  /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
  def llama_sampler_init_penalties(
      penalty_last_n: CInt,
      penalty_repeat: CFloat,
      penalty_freq: CFloat,
      penalty_present: CFloat
  ): Sampler

  ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
  def llama_sampler_init_dry(
      vocab: Vocab,
      n_ctx_train: CInt,
      dry_multiplier: CFloat,
      dry_base: CFloat,
      dry_allowed_length: CInt,
      dry_penalty_last_n: CInt,
      seq_breakers: Ptr[Ptr[CChar]],
      num_breakers: SizeT
  ): Sampler

  def llama_sampler_init_logit_bias(
      n_vocab: CInt,
      n_logit_bias: CInt,
      logit_bias: Ptr[LogitBias]
  ): Sampler

  // this sampler is meant to be used for fill-in-the-middle infilling
  // it's supposed to be used after top_k + top_p sampling
  //
  // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
  // 2. combine probs of tokens that have the same prefix
  //
  // example:
  //
  // - before:
  //   "hel":   0.5
  //   "hell":  0.2
  //   "hello": 0.1
  //   "dummy": 0.1
  //
  // - after:
  //   "hel":   0.8
  //   "dummy": 0.1
  //
  // 3. discard non-EOG tokens with low prob
  // 4. if no tokens are left -> pick EOT
  def llama_sampler_init_infill(vocab: Vocab): Sampler

  // Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
  def llama_sampler_get_seed(smpl: Sampler): CInt

  /// @details Sample and accept a token from the idx-th output of the last evaluation
  //
  // Shorthand for:
  //    const auto * logits = llama_get_logits_ith(ctx, idx);
  //    llama_token_data_array cur_p = { ... init from logits ... };
  //    llama_sampler_apply(smpl, &cur_p);
  //    auto token = cur_p.data[cur_p.selected].id;
  //    llama_sampler_accept(smpl, token);
  //    return token;
  // Returns the sampled token
  def llama_sampler_sample(smpl: Sampler, ctx: Ctx, idx: CInt): Token

  // Model split

  /// @details Build a split GGUF final path for this chunk.
  ///          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
  //  Returns the split_path length.
  def llama_split_path(
      split_path: Ptr[CChar],
      maxlen: SizeT,
      path_prefix: Ptr[CChar],
      split_no: CInt,
      split_count: CInt
  ): CInt

  /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
  ///          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
  //  Returns the split_prefix length.
  def llama_split_prefix(
      split_prefix: Ptr[CChar],
      maxlen: SizeT,
      split_path: Ptr[CChar],
      split_no: CInt,
      split_count: CInt
  ): CInt

  // Print system information
  def llama_print_system_info(): Ptr[CChar]
