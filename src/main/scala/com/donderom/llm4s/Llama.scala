package com.donderom.llm4s

import fr.hammons.slinc.types.*
import fr.hammons.slinc.*

case class llama_context_params(
    seed: Int,
    n_ctx: Int,
    n_batch: Int,
    n_gpu_layers: Int,
    main_gpu: Int,
    tensor_split: SetSizeArray[Float, 1],
    progress_callback: Ptr[(Float, Unit) => Unit],
    progress_callback_user_data: Ptr[Unit],
    low_vram: Byte,
    f16_kv: Byte,
    logits_all: Byte,
    vocab_only: Byte,
    use_mmap: Byte,
    use_mlock: Byte,
    embedding: Byte
) derives Struct

case class llama_token_data(
    id: Int,
    logit: Float,
    p: Float
) derives Struct

case class llama_token_data_array(
    data: Ptr[llama_token_data],
    size: SizeT,
    sorted: Byte
) derives Struct

enum LlamaFtype(val code: Int):
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

case class llama_model_quantize_params(
    nthread: Int,
    ftype: Int,
    allow_requantize: Byte,
    quantize_output_tensor: Byte
) derives Struct

trait Llama derives FSet:
  type LlamaToken = CInt
  type Ctx = Ptr[Any]
  type Model = Ptr[Any]

  def llama_context_default_params(): llama_context_params
  def llama_model_quantize_default_params(): llama_model_quantize_params

  def llama_mmap_supported(): Byte
  def llama_mlock_supported(): Byte

  def llama_init_backend(): Unit

  def llama_time_us(): CInt

  def llama_load_model_from_file(
      path_model: Ptr[CChar],
      params: llama_context_params
  ): Model

  def llama_free_model(model: Model): Unit

  def llama_new_context_with_model(
      model: Model,
      params: llama_context_params
  ): Ctx

  def llama_free(ctx: Ctx): Unit

  def llama_model_quantize(
      fname_inp: Ptr[CChar],
      fname_out: Ptr[CChar],
      params: Ptr[llama_model_quantize_params]
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
  ): Byte

  def llama_save_session_file(
      ctx: Ctx,
      path_session: Ptr[CChar],
      tokens: Ptr[LlamaToken],
      n_token_out: SizeT
  ): Byte

  def llama_eval(
      ctx: Ctx,
      tokens: Ptr[LlamaToken],
      n_tokens: CInt,
      n_past: CInt,
      n_threads: CInt
  ): CInt

  def llama_eval_export(ctx: Ctx, fname: Ptr[CChar]): CInt

  def llama_tokenize(
      ctx: Ctx,
      text: Ptr[CChar],
      tokens: Ptr[LlamaToken],
      n_max_tokens: Int,
      add_bos: Byte
  ): CInt

  def llama_n_vocab(ctx: Ctx): CInt
  def llama_n_ctx(ctx: Ctx): CInt
  def llama_n_embd(ctx: Ctx): CInt

  def llama_get_vocab(
      ctx: Ctx,
      strings: Ptr[Ptr[CChar]],
      scores: Ptr[Float],
      capacity: Int
  ): CInt

  def llama_get_logits(ctx: Ctx): Ptr[CFloat]

  def llama_get_embeddings(ctx: Ctx): Ptr[CFloat]

  def llama_token_to_str(ctx: Ctx, token: LlamaToken): Ptr[CChar]

  def llama_token_bos(): LlamaToken
  def llama_token_eos(): LlamaToken
  def llama_token_nl(): LlamaToken

  def llama_sample_repetition_penalty(
      ctx: Ctx,
      candidates: Ptr[llama_token_data_array],
      last_tokens: Ptr[LlamaToken],
      last_tokens_size: SizeT,
      penalty: Float
  ): Unit

  def llama_sample_frequency_and_presence_penalties(
      ctx: Ctx,
      candidates: Ptr[llama_token_data_array],
      last_tokens: Ptr[LlamaToken],
      last_tokens_size: SizeT,
      alpha_frequency: Float,
      alpha_presence: Float
  ): Unit

  def llama_sample_softmax(
      ctx: Ctx,
      candidates: Ptr[llama_token_data_array]
  ): Unit

  def llama_sample_top_k(
      ctx: Ctx,
      candidates: Ptr[llama_token_data_array],
      k: CInt,
      min_keep: SizeT
  ): Unit

  def llama_sample_top_p(
      ctx: Ctx,
      candidates: Ptr[llama_token_data_array],
      p: CFloat,
      min_keep: SizeT
  ): Unit

  def llama_sample_tail_free(
      ctx: Ctx,
      candidates: Ptr[llama_token_data_array],
      z: CFloat,
      min_keep: SizeT
  ): Unit

  def llama_sample_typical(
      ctx: Ctx,
      candidates: Ptr[llama_token_data_array],
      p: CFloat,
      min_keep: SizeT
  ): Unit

  def llama_sample_temperature(
      ctx: Ctx,
      candidates: Ptr[llama_token_data_array],
      temp: CFloat
  ): Unit

  def llama_sample_token_mirostat(
      ctx: Ctx,
      candidates: Ptr[llama_token_data_array],
      tau: CFloat,
      eta: CFloat,
      m: CInt,
      mu: Ptr[CFloat]
  ): LlamaToken

  def llama_sample_token_mirostat_v2(
      ctx: Ctx,
      candidates: Ptr[llama_token_data_array],
      tau: CFloat,
      eta: CFloat,
      mu: Ptr[CFloat]
  ): LlamaToken

  def llama_sample_token_greedy(
      ctx: Ctx,
      candidates: Ptr[llama_token_data_array]
  ): LlamaToken

  def llama_sample_token(
      ctx: Ctx,
      candidates: Ptr[llama_token_data_array]
  ): LlamaToken

  def llama_print_timings(ctx: Ctx): Unit
  def llama_reset_timings(ctx: Ctx): Unit

  def llama_print_system_info(): Ptr[CChar]
