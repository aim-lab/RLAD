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
from tqdm import tqdm
import PIL
from PIL import Image
import os
from safetensors.torch import load_model, save_model

def eval():
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

    model = model.bfloat16()

    data_module = DataModule(model_args)
    try:
        print("loading weights: ", data_module.data_config['load_weights_from'])
        load_model(model, data_module.data_config['load_weights_from'])
    except:
        print("No weights found")
    prediction_path_save = data_module.data_config['generation_path_save']

    n_gen = data_module.data_config.get("n_gen_per_samples",15)
    CD_cond = data_module.data_config.get("CD_cond",True)
    L_cond = data_module.data_config.get("L_cond",True)


    ds = data_module.test_dataset
    collator = data_module.get_data_collator()
    n = 0
    for element in ds:
        n_gen_2 = 1
        guidance_scale = data_module.data_config.get('guidance_scale', (1.1, 1.5))
        inputs_ = [element for i in range(n_gen)]
        inputs = collator(inputs_)
        print(f"working on {prediction_path_save}_0/{inputs_[0]['dataset_name']}/{inputs_[0]['image_name']}", n)

        for j in range(n_gen_2):

            with torch.no_grad():
                out = model.generate(num_gen = n_gen,
                                     images_bv = inputs['images_bv'].to("cuda").bfloat16(),
                                     images_odin = inputs['images_odin'].to("cuda").bfloat16() if CD_cond else None,
                                     images_lesion = inputs['images_odin'].to("cuda").bfloat16() if L_cond else None,
                                     num_inference_steps = 50,
                                     guidance_scale=guidance_scale,
                                     # strict = False
                                     )
            os.makedirs(f"{prediction_path_save}_orig_image/{inputs_[0]['dataset_name']}", exist_ok=True)
            seg = PIL.Image.fromarray(np.array((0.5 + (inputs['images'][0].permute(1, 2, 0) * 0.5)) * 255, dtype=np.uint8))
            seg.save(f"{prediction_path_save}_orig_image/{inputs_[0]['dataset_name']}/{inputs_[0]['image_name']}")

            for i in range(out.shape[0]):
                # Create a unique index for each image generated, incorporating both j and i
                idx = (j * out.shape[0]) + i  # Keeps a linear increase with each iteration
                output_dir = f"{prediction_path_save}_{idx}/{inputs_[0]['dataset_name']}"
                os.makedirs(output_dir, exist_ok=True)
                im = PIL.Image.fromarray(np.array(out[i] * 255, dtype=np.uint8))
                im.save(f"{output_dir}/{inputs_[0]['image_name']}")


if __name__ == "__main__":
    print(os.getcwd())
    # Seed can be changed for new randomness
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    logging.basicConfig(level=logging.INFO)
    eval()



