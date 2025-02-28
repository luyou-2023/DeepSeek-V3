'''
https://zhuanlan.zhihu.com/p/24271132165
'''

del model

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = hf_token,
)


FastLanguageModel.for_training(model)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,    # 负责 分词（保证训练数据格式正确）
    train_dataset=dataset,    # 训练数据集（预处理后的 Hugging Face Dataset）
    dataset_text_field="text",    # 训练数据的 文本字段（例如 "text"）
    max_seq_length=max_seq_length,    # 最大序列长度（如 2048）
    dataset_num_proc=2,    # 数据预处理的 CPU 进程数（加速数据加载）
    args=TrainingArguments(    # 训练参数
        per_device_train_batch_size=2,    # 每块 GPU 的 batch size（小显存建议 2-4）
        gradient_accumulation_steps=4,    # 梯度累积，相当于 batch size 放大 4x
        warmup_steps=5,    # 预热步数，前 5 步学习率从 0 线性增加
        max_steps=60,    # 训练 60 步（适用于快速测试）
        learning_rate=2e-4,    # 默认 2e-4（LoRA 一般 2e-4 ~ 1e-5）
        fp16=not is_bfloat16_supported(),    # NVIDIA A100/H100/4090 支持，训练更稳定
        bf16=is_bfloat16_supported(),    # T4/V100/2080 不支持 BF16，改用 FP16
        logging_steps=10,    # 每 10 步打印训练日志
        optim="adamw_8bit",    # 使用 8-bit AdamW，减少显存占用
        weight_decay=0.01,    # L2 正则化，防止过拟合
        lr_scheduler_type="linear",    # 线性学习率衰减
        seed=3407,    # 设定随机种子，保证实验可复现
        output_dir="outputs",    # 模型 & 训练日志保存路径
    ),
)

trainer_stats = trainer.train()
