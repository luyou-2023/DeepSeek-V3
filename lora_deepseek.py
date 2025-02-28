import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW


class LoRALayer(nn.Module):
    """
    LoRA低秩适应层。这个层将被用于模型中的某些全连接层。
    """
    def __init__(self, input_dim, output_dim, rank=8):
        super(LoRALayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        # LoRA参数初始化
        self.A = nn.Parameter(torch.randn(input_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, output_dim))

    def forward(self, x):
        # LoRA低秩适应
        return torch.matmul(torch.matmul(x, self.A), self.B)


class LoRAAdaptedModel(nn.Module):
    """
    将LoRA层集成到现有模型中的封装类。
    在此示例中，我们将LoRA层插入到BERT模型的部分层中。
    """
    def __init__(self, original_model, rank=8):
        super(LoRAAdaptedModel, self).__init__()
        self.model = original_model
        self.rank = rank

        # 替换Transformer中的某些线性层为LoRA层
        self.model.encoder.layer[0].attention.self.query = LoRALayer(
            self.model.config.hidden_size, self.model.config.hidden_size, rank=self.rank
        )
        self.model.encoder.layer[0].attention.self.key = LoRALayer(
            self.model.config.hidden_size, self.model.config.hidden_size, rank=self.rank
        )
        self.model.encoder.layer[0].attention.self.value = LoRALayer(
            self.model.config.hidden_size, self.model.config.hidden_size, rank=self.rank
        )

    def forward(self, input_ids):
        return self.model(input_ids)


def main():
    # 加载预训练模型
    model_name = 'bert-base-uncased'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 使用LoRA适应模型
    lora_model = LoRAAdaptedModel(model, rank=8)

    # 定义优化器
    optimizer = AdamW(lora_model.parameters(), lr=5e-5)

    # 示例输入数据
    text = "This is an example sentence."
    inputs = tokenizer(text, return_tensors="pt")

    # 前向传播
    outputs = lora_model(**inputs)
    print(outputs)

    # 训练循环示例（这里只做了简单的前向传播）
    optimizer.zero_grad()
    loss = outputs.loss
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    main()
