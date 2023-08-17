package com.donderom.llm4s

import fr.hammons.slinc.types.*
import fr.hammons.slinc.{FSet, Ptr, Struct, Transform}

object Llama:
  final case class ContextParams(
      seed: CInt,
      n_ctx: CInt,
      n_batch: CInt,
      n_gqa: CInt,
      rms_norm_eps: CFloat,
      n_gpu_layers: CInt,
      main_gpu: CInt,
      tensor_split: Ptr[CFloat],
      rope_freq_base: CFloat,
      rope_freq_scale: CFloat,
      progress_callback: Ptr[(CFloat, Unit) => Unit],
      progress_callback_user_data: Ptr[Unit],
      low_vram: CBool,
      mul_mat_q: CBool,
      f16_kv: CBool,
      logits_all: CBool,
      vocab_only: CBool,
      use_mmap: CBool,
      use_mlock: CBool,
      embedding: CBool
  ) derives Struct

  final case class TokenData(id: CInt, logit: CFloat, p: CFloat) derives Struct

  final case class TokenDataArray(
      data: Ptr[TokenData],
      size: SizeT,
      sorted: CBool
  ) derives Struct

  enum LlamaFtype(val code: CInt):
    case LLAMA_FTYPE_ALL_F32 extends LlamaFtype(0)
    case LLAMA_FTYPE_MOSTLY_F16 extends LlamaFtype(1)
    case LLAMA_FTYPE_MOSTLY_Q4_0 extends LlamaFtype(2)
    case LLAMA_FTYPE_MOSTLY_Q4_1 extends LlamaFtype(3)
    case LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 extends LlamaFtype(4)
    case LLAMA_FTYPE_MOSTLY_Q8_0 extends LlamaFtype(7)
    case LLAMA_FTYPE_MOSTLY_Q5_0 extends LlamaFtype(8)
    case LLAMA_FTYPE_MOSTLY_Q5_1 extends LlamaFtype(9)
    case LLAMA_FTYPE_MOSTLY_Q2_K extends LlamaFtype(10)
    case LLAMA_FTYPE_MOSTLY_Q3_K_S extends LlamaFtype(11)
    case LLAMA_FTYPE_MOSTLY_Q3_K_M extends LlamaFtype(12)
    case LLAMA_FTYPE_MOSTLY_Q3_K_L extends LlamaFtype(13)
    case LLAMA_FTYPE_MOSTLY_Q4_K_S extends LlamaFtype(14)
    case LLAMA_FTYPE_MOSTLY_Q4_K_M extends LlamaFtype(15)
    case LLAMA_FTYPE_MOSTLY_Q5_K_S extends LlamaFtype(16)
    case LLAMA_FTYPE_MOSTLY_Q5_K_M extends LlamaFtype(17)
    case LLAMA_FTYPE_MOSTLY_Q6_K extends LlamaFtype(18)

  given Transform[LlamaFtype, CInt](
    cint => LlamaFtype.fromOrdinal(cint),
    ftype => ftype.code
  )

  final case class ModelQuantizeParams(
      nthread: CInt,
      ftype: LlamaFtype,
      allow_requantize: CBool,
      quantize_output_tensor: CBool
  ) derives Struct

  enum LlamaGretype:
    case LLAMA_GRETYPE_END,
      LLAMA_GRETYPE_ALT,
      LLAMA_GRETYPE_RULE_REF,
      LLAMA_GRETYPE_CHAR,
      LLAMA_GRETYPE_CHAR_NOT,
      LLAMA_GRETYPE_CHAR_RNG_UPPER,
      LLAMA_GRETYPE_CHAR_ALT

  given Transform[LlamaGretype, CInt](
    cint => LlamaGretype.fromOrdinal(cint),
    gretype => gretype.ordinal
  )

  final case class GrammarElement(elementType: LlamaGretype, value: CInt)
      derives Struct

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

  type LlamaToken = CInt
  type Ctx = Ptr[Any]
  type Model = Ptr[Any]
  type Grammar = Ptr[Any]

trait Llama derives FSet:
  import Llama.*

  def llama_max_devices(): CInt

  def llama_context_default_params(): ContextParams
  def llama_model_quantize_default_params(): ModelQuantizeParams

  def llama_mmap_supported(): CBool
  def llama_mlock_supported(): CBool

  def llama_backend_init(numa: CBool): Unit

  def llama_backend_free(): Unit

  def llama_time_us(): CInt

  def llama_load_model_from_file(
      path_model: Ptr[CChar],
      params: ContextParams
  ): Model

  def llama_free_model(model: Model): Unit

  def llama_new_context_with_model(
      model: Model,
      params: ContextParams
  ): Ctx

  def llama_free(ctx: Ctx): Unit

  def llama_model_quantize(
      fname_inp: Ptr[CChar],
      fname_out: Ptr[CChar],
      params: Ptr[ModelQuantizeParams]
  ): CInt

  def llama_model_apply_lora_from_file(
      model: Model,
      path_lora: Ptr[CChar],
      path_base_model: Ptr[CChar],
      n_threads: CInt
  ): CInt

  def llama_get_kv_cache_token_count(ctx: Ctx): CInt

  def llama_set_rng_seed(ctx: Ctx, seed: CInt): Unit

  def llama_get_state_size(ctx: Ctx): SizeT

  def llama_copy_state_data(ctx: Ctx, dst: Ptr[Byte]): SizeT

  def llama_set_state_data(ctx: Ctx, src: Ptr[Byte]): SizeT

  def llama_load_session_file(
      ctx: Ctx,
      path_session: Ptr[CChar],
      tokens_out: Ptr[LlamaToken],
      n_token_capacity: SizeT,
      n_token_count_out: SizeT
  ): CBool

  def llama_save_session_file(
      ctx: Ctx,
      path_session: Ptr[CChar],
      tokens: Ptr[LlamaToken],
      n_token_out: SizeT
  ): CBool

  def llama_eval(
      ctx: Ctx,
      tokens: Ptr[LlamaToken],
      n_tokens: CInt,
      n_past: CInt,
      n_threads: CInt
  ): CInt

  def llama_eval_embd(
      ctx: Ctx,
      embd: Ptr[CFloat],
      n_tokens: CInt,
      n_past: CInt,
      n_threads: CInt
  ): CInt

  def llama_eval_export(ctx: Ctx, fname: Ptr[CChar]): CInt

  def llama_tokenize(
      ctx: Ctx,
      text: Ptr[CChar],
      tokens: Ptr[LlamaToken],
      n_max_tokens: CInt,
      add_bos: CBool
  ): CInt

  def llama_tokenize_with_model(
      model: Model,
      text: Ptr[Char],
      tokens: Ptr[LlamaToken],
      n_max_tokens: CInt,
      add_bos: CBool
  ): CInt

  def llama_n_vocab(ctx: Ctx): CInt
  def llama_n_ctx(ctx: Ctx): CInt
  def llama_n_embd(ctx: Ctx): CInt

  def llama_n_vocab_from_model(model: Model): CInt
  def llama_n_ctx_from_model(model: Model): CInt
  def llama_n_embd_from_model(model: Model): CInt

  def llama_get_vocab(
      ctx: Ctx,
      strings: Ptr[Ptr[CChar]],
      scores: Ptr[CFloat],
      capacity: CInt
  ): CInt

  def llama_get_vocab_from_model(
      model: Model,
      strings: Ptr[Ptr[CChar]],
      scores: Ptr[CFloat],
      capacity: CInt
  ): CInt

  def llama_get_logits(ctx: Ctx): Ptr[CFloat]

  def llama_get_embeddings(ctx: Ctx): Ptr[CFloat]

  def llama_token_to_str(ctx: Ctx, token: LlamaToken): Ptr[CChar]
  def llama_token_to_str_with_model(model: Model, token: LlamaToken): Ptr[CChar]

  def llama_token_bos(): LlamaToken
  def llama_token_eos(): LlamaToken
  def llama_token_nl(): LlamaToken

  def llama_grammar_init(
      rules: Ptr[Ptr[GrammarElement]],
      n_rules: SizeT,
      start_rule_index: SizeT
  ): Grammar
  def llama_grammar_free(grammar: Grammar): Unit

  def llama_sample_repetition_penalty(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      last_tokens: Ptr[LlamaToken],
      last_tokens_size: SizeT,
      penalty: CFloat
  ): Unit

  def llama_sample_frequency_and_presence_penalties(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      last_tokens: Ptr[LlamaToken],
      last_tokens_size: SizeT,
      alpha_frequency: CFloat,
      alpha_presence: CFloat
  ): Unit

  def llama_sample_classifier_free_guidance(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      guidance_ctx: Ctx,
      scale: CFloat
  ): Unit

  def llama_sample_softmax(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray]
  ): Unit

  def llama_sample_top_k(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      k: CInt,
      min_keep: SizeT
  ): Unit

  def llama_sample_top_p(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      p: CFloat,
      min_keep: SizeT
  ): Unit

  def llama_sample_tail_free(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      z: CFloat,
      min_keep: SizeT
  ): Unit

  def llama_sample_typical(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      p: CFloat,
      min_keep: SizeT
  ): Unit

  def llama_sample_temperature(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      temp: CFloat
  ): Unit

  def llama_sample_grammar(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      grammar: Grammar
  ): Unit

  def llama_sample_token_mirostat(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      tau: CFloat,
      eta: CFloat,
      m: CInt,
      mu: Ptr[CFloat]
  ): LlamaToken

  def llama_sample_token_mirostat_v2(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray],
      tau: CFloat,
      eta: CFloat,
      mu: Ptr[CFloat]
  ): LlamaToken

  def llama_sample_token_greedy(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray]
  ): LlamaToken

  def llama_sample_token(
      ctx: Ctx,
      candidates: Ptr[TokenDataArray]
  ): LlamaToken

  def llama_grammar_accept_token(
      ctx: Ctx,
      grammar: Grammar,
      token: LlamaToken
  ): Unit

  def llama_get_timings(ctx: Ctx): Timings
  def llama_print_timings(ctx: Ctx): Unit
  def llama_reset_timings(ctx: Ctx): Unit

  def llama_print_system_info(): Ptr[CChar]
