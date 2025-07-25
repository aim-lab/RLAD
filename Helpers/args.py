from dataclasses import dataclass, field
from typing import Optional
import transformers

@dataclass
class ModelArguments:
    model_config_path: str = field(default=None)
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    evaluation_strategy: str = field(default="epoch")
    num_train_epochs: float = field(default=100)
    bf16: bool = field(default=False)
