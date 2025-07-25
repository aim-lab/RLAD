from Helpers.utils import load_yaml
from Data.RetinaDataset import Retina_dataset
import torch
from .collator import DataCollator
import math
class DataModule:

    def __init__(self, data_args):
        self.data_args = data_args
        self._train_dataset = None
        self._eval_dataset = None
        self._test_dataset = None
        self._gen_support_dataset = None
        self._data_colator = None

        if hasattr(data_args,'model_config_path'):
            self.data_config = load_yaml(data_args.model_config_path)

        try:
            self.image_size = int(self.data_config.get("resolution", 224))
        except:
            self.image_size = self.data_config.get("resolution", 224)
        print("Using resolution", self.image_size)

        self.pad = int(self.data_config.get("pad", 0))
        print("Padding to: ", self.pad)


        self.root_dir = self.data_config.get("root_dir")
        print("Train Root experience dir: ", self.root_dir)

        self.val_root_dir = self.data_config.get("val_root_dir", self.root_dir)
        self.test_root_dir = self.data_config.get("test_root_dir", self.root_dir)
        self.gen_root_dir = self.data_config.get("gen_root_dir", self.root_dir)

        print(f"val/test Root experience dir: {self.val_root_dir}/{self.test_root_dir}")


        self.segmentation_supervision = self.data_config.get("segmentation_supervision", True)
        print("segmentation supervision: ", self.segmentation_supervision)


        self.special_bv_res = self.data_config.get("special_bv_res", None)
        print("special_bv_res: ", self.special_bv_res)

        self.sampling = self.data_config.get("sampling",None)


    @property
    def train_dataset(self):
        if self._train_dataset is None:
            datasets = []
            for dataset in self.data_config.get("train_datasets"):
                datasets.append(Retina_dataset(
                    dataset,
                    self.root_dir,
                    "train",
                    self.image_size,
                    special_bv_res = self.special_bv_res
                ))

            print(f"Datasets used for train, {len(datasets)}, {datasets}")
            self._train_dataset = torch.utils.data.ConcatDataset(datasets)
            print(f"Datasets used for train, containing a total of {len(self._train_dataset)} samples")

            if self.sampling == "proportional_sqrt":
                print("Using proportional sqrt sampling")
                # Calculate weights based on the square root of dataset sizes
                dataset_sizes = [len(dataset) for dataset in datasets]
                total_size = sum(math.sqrt(size) for size in dataset_sizes)

                # Create a list of weights for each element in the concatenated dataset
                weights = []
                for size in dataset_sizes:
                    weight = math.sqrt(size) / total_size
                    weights.extend([weight] * size)

                # Create a WeightedRandomSampler
                self.sampler = torch.utils.data.WeightedRandomSampler(
                    weights, num_samples=sum(dataset_sizes), replacement=True
                )

            if self.sampling == "proportional":
                print("Using proportional sampling")
                # Calculate weights for each dataset
                dataset_sizes = [len(dataset) for dataset in datasets]
                total_size = sum(dataset_sizes)

                # Create a list of weights for each element in the concatenated dataset
                weights = []
                for size in dataset_sizes:
                    weight = size / total_size
                    weights.extend([weight] * size)

                # Create a WeightedRandomSampler
                self.sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=total_size, replacement=True)

        return self._train_dataset

    @property
    def eval_dataset(self):
        if self._eval_dataset is None:
            datasets = []
            for dataset in self.data_config.get("val_dataset"):
                datasets.append(Retina_dataset(
                    dataset,
                    self.val_root_dir,
                    "val",
                    self.image_size,
                    special_bv_res = self.special_bv_res

                ))
            print(f"Datasets used for eval, {len(datasets)}, {datasets}")
            self._eval_dataset = torch.utils.data.ConcatDataset(datasets)
            print(f"Datasets used for eval, containing a total of {len(self._eval_dataset)} samples")
        return self._eval_dataset

    @property
    def test_dataset(self):
        if self._test_dataset is None:
            datasets = []
            for dataset in self.data_config.get("test_dataset"):
                datasets.append(Retina_dataset(
                    dataset,
                    self.test_root_dir,
                    "test",
                    self.image_size,
                    special_bv_res=self.special_bv_res
                ))

            print(f"Datasets used for test, {len(datasets)}, {datasets}")
            self._test_dataset = torch.utils.data.ConcatDataset(datasets)
            print(f"Datasets used for test, containing a total of {len(self._test_dataset)} samples")
        return self._test_dataset



    @property
    def gen_support_dataset(self):
        if self._gen_support_dataset is None:
            datasets = []
            for dataset in self.data_config.get("gen_support_dataset"):
                datasets.append(Retina_dataset(
                    dataset,
                    self.gen_root_dir,
                    "test",
                    self.image_size,
                    special_bv_res=self.special_bv_res
                ))

            print(f"Datasets used for test, {len(datasets)}, {datasets}")
            self._gen_support_dataset = torch.utils.data.ConcatDataset(datasets)
            print(f"Datasets used for gen_support_dataset, containing a total of {len(self._gen_support_dataset)} samples")
        return self._gen_support_dataset

    def get_data_collator(self):
        return DataCollator(self.pad)

    @property
    def data_collator(self):
        return self.get_data_collator()

    def to_dict(self,do_train = True):
        ret = dict(
            data_collator = self.data_collator,
            eval_dataset = self.eval_dataset
        )
        if do_train:
            ret['train_dataset'] = self.train_dataset
            if hasattr(self,"sampler"):
                ret['sampler'] = self.sampler
        return ret





