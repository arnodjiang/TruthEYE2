# TruthEYE2: Agentic MLLMs for Unified Synthetic Image Forensics via Programmatic Visual Reasoning

<p align="center">
<strong>Changjiang Jiang</strong><sup>*</sup>, Mingqi Fang<sup>*</sup>, Bo Du, Xuekang Zhu, Zixuan Zhang, Chenfan Qu, Zhenming Wang, Jingjing Liu<sup>#</sup>, Jian Liu<sup>#</sup>.
</p>

<p align="center">
<small>
**Ant Group**
* Equal Contribution
# Corresponding author
</small>
</p>

> TruthEYE2 is an agentic multimodal large language model (MLLM) framework for unified synthetic image forensics, supporting image-level authenticity detection, fine-grained forgery localization, and interpretable reasoning in an end-to-end manner. Unlike prior tool-based approaches that rely on a fixed and predefined tool registry, TruthEYE2 adopts Programmatic Visual Reasoning (PVR): the model directly generates executable image-operation code (e.g., rotation, cropping) during reasoning, enabling flexible, compositional, and scalable visual forensics.
> This repository provides the implementation of TruthEYE2, including training pipelines, inference code, and the TruthEYETool-50K dataset introduced in our paper.

## Training Pipeline

TruthEYE2 is trained with a three-stage training paradigm, base on Qwen2.5-VL:

Stage 1: Cold Start (iMCoT Alignment)

- Download CodeVision Dataset
- Preprocess CodeVision Dataset, delete failure samples

运行：

```bash
swift sft \
    --model "/Qwen2.5-VL-7B-Instruct" \
    --model_type "qwen2_5_vl" \
    --train_type full \
    --dataset "./codevision_sft_swift.jsonl" \ # modify data
    --max_length 4096 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner true \
    --save_steps 20 \
    --save_total_limit 1 \
    --logging_steps 2 \
    --output_dir /sft1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --attn_impl flash_attention_2 \
    --deepspeed zero2 \
    --repetition_penalty 1.05 \
    --response_prefix "<think>\n" \
    --gradient_checkpointing true \
    --loss_scale qwen
```

Stage 2: Reinforcement Learning for FIDL


运行：

```bash
swift rlhf \
    --rlhf_type grpo \
    --model /sft1 \
    --external_plugins ./plugs.py \
    --reward_funcs trutheyes2_reward \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 2 \
    --vllm_data_parallel_size 4 \
    --train_type full \
    --sleep_level 1 \
    --torch_dtype bfloat16 \
    --dataset "./rl_train.jsonl" \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --soft_cache_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --output_dir $out_path \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --num_generations 8 \
    --temperature 1.05 \
    --epsilon_high 0.6 \
    --deepspeed zero3 \
    --log_completions true \
    --async_generate false \
    --offload_optimizer true \
    --offload_model true \
    --repetition_penalty 1.10 \
    --beta 0.04 \
    --attn_impl flash_attn \
    --overlong_filter true \
    --truncation_strategy delete \
    --dynamic_sample true \
    --stop_words "<answer>real</answer>" "<answer>fake</answer>" "<think>\n<think>" ">\n>" \
    --response_prefix "<think>"
```

Stage 3: Reinforcement Learning on Hard Cases

```bash
swift rlhf \
    --rlhf_type grpo \
    --model /sft2 \
    --external_plugins ./plugs.py \
    --reward_funcs trutheyes2_reward \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 2 \
    --vllm_data_parallel_size 4 \
    --train_type full \
    --sleep_level 1 \
    --torch_dtype bfloat16 \
    --dataset "./rl_train_stage3.jsonl" \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --soft_cache_length 1024 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-6 \
    --epsilon_high 0.6 \
    --gradient_accumulation_steps 1 \
    --save_strategy 'steps' \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --output_dir $out_path \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --num_generations 8 \
    --temperature 1.05 \
    --deepspeed zero3 \
    --log_completions true \
    --async_generate false \
    --offload_optimizer true \
    --offload_model true \
    --repetition_penalty 1.10 \
    --beta 0.04 \
    --attn_impl flash_attn \
    --overlong_filter true \
    --truncation_strategy delete \
    --dynamic_sample true \
    --stop_words "<answer>real</answer>" "<answer>fake</answer>" "<think>\n<think>" ">\n>" \
    --response_prefix "<think>"
```


Ethical Considerations
- All data used are sourced from publicly available datasets for research purposes
- No private, sensitive, or personally identifiable information is included