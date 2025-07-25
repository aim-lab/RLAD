import logging
import os
import sys
import yaml
import transformers
from transformers import PreTrainedModel
from Helpers.args import TrainingArguments,ModelArguments
from Helpers.utils import load_yaml, get_object_from_path
import torch
from Data.data_module import DataModule
from Trainer.trainer import SegTrainer
import numpy as np
import random



def train():
    parser = transformers.HfArgumentParser((ModelArguments,TrainingArguments))
    model_args,training_args = parser.parse_args_into_dataclasses()
    save_dir = training_args.output_dir
    print("Experience will be saved in ",save_dir)
    logger = logging.getLogger()
    logger.info(
        f"training args {training_args}, model_args {model_args}"
    )
    num_gpus = 1
    if torch.distributed.is_initialized():
        num_gpus = torch.distributed.get_world_size()

    model_class = load_yaml(model_args.model_config_path).get('model_class',None)

    model: PreTrainedModel = get_object_from_path(model_class)(model_args)

    model.to("cuda")

    data_module = DataModule(model_args)

    is_multilabel = data_module.data_config.get("multilabel", False)

    # ds = data_module.train_dataset
    # collator = data_module.get_data_collator()
    # items = [ds[0], ds[1], ds[2]]
    # inputs = collator(items)
    # model(inputs['images'].cuda())
    # # im = np.array((item[1]['images'] * 0.5 + 0.5)* 255, dtype=np.uint8).transpose((1, 2, 0))
    # # PIL.Image.fromarray(im).save('test.png')

    trainer = SegTrainer(model = model,
                        args = training_args,
                        is_multilabel=is_multilabel,
                        **data_module.to_dict())

    trainer.train()

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    logging.basicConfig(level=logging.INFO)
    train()



