import random
from torch.utils.data import Dataset
import PIL
from PIL import Image
import numpy as np
import pathlib
from torchvision import transforms
import cv2
datasets_with_segmentation = ["UZLF_TRAIN","GRAPE","MAGREB","MESSIDOR", "PAPILA", "LES2", "UZLF_VAL", "AVWIDE", "DRIVE_train","DRIVE_test","IOSTAR","UZLF_VAL","UZLF_TEST","INSPIRE","FIVES","LESAV","PARAGUAY","HRF","RVD"]
datasets_with_segmentation += ["UZLF_TRAIN","GRAPE_TRAIN","MAGREB_TRAIN","MESSIDOR_TRAIN", "PAPILA_TRAIN", "LES2_TRAIN", "UZLF_VAL", "AVWIDE", "DRIVE_train","DRIVE_test","IOSTAR","UZLF_VAL","UZLF_TEST","INSPIRE","FIVES","LESAV","PARAGUAY","HRF","RVD"]

class Retina_dataset(Dataset):
    def __init__(self, base_dir, root_dir, split, img_size, special_bv_res = None):
        self.split = split
        self.data_dir = pathlib.Path(root_dir) / base_dir
        self.sample_list = list((self.data_dir / "images").glob("*.png")) + list((self.data_dir / "images").glob("*.jpg")) +  list((self.data_dir / "images").glob("*.jpeg"))
        if len(self.sample_list) == 0:
            self.sample_list = list(self.data_dir.glob("*.png")) + list(self.data_dir.glob("*.jpg")) +  list(self.data_dir.glob("*.jpeg"))
        self.img_size = img_size
        self.dataset_name = base_dir
        self.special_bv_res = special_bv_res

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        data_path = self.sample_list[idx]
        img_orig = PIL.Image.open(data_path).convert('RGB')
        img_orig_size = img_orig.size

        ## Crop and pad image to square if needed
        if self.dataset_name in datasets_with_segmentation:
            img_orig = Image.fromarray(pad_to_square(np.array(img_orig)))
        else:
            img_orig = Image.fromarray(pad_to_square(crop_black(np.array(img_orig))))
        #Resize the image to input resolution and blood vessel segmenter input resolution
        image = img_orig.resize((self.img_size, self.img_size))
        image_bv = img_orig.resize((self.special_bv_res, self.special_bv_res))



        if self.split == "train":
            # Apply the same random horizontal and vertical flip
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                image_bv = transforms.functional.hflip(image_bv)

            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                image_bv = transforms.functional.vflip(image_bv)


        # Convert PIL Images to tensors
        image = transforms.ToTensor()(image)
        image_bv = transforms.ToTensor()(image_bv)


        # Normalize
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        normalize_lunet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



        image_odin = normalize_lunet(image)
        image = normalize(image)
        image_bv = normalize_lunet(image_bv)


        sample = {'images': image,
                  'images_bv': image_bv,
                  'dataset_name': self.dataset_name,
                  'image_name': str(data_path).split('/')[-1],
                  'img_orig_size': img_orig_size,
                  'images_odin': image_odin,
                  }
        return sample

def crop_black(img,
               tol=7):
    '''
    Perform automatic crop of black areas. Origin: https://www.kaggle.com/code/sayedmahmoud/diabetic-retinopathy-detection
    '''

    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        if (check_shape == 0):
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
            return img

def pad_to_square(image):
    dim = max(image.shape[0],image.shape[1])
    new_image = np.zeros((dim,dim,3),dtype = image.dtype)
    if image.shape[0] == image.shape[1]:
        return image
    if dim == image.shape[0]:
        new_image[:,(dim - image.shape[1])//2:-(dim - image.shape[1])//2 ] = image
    else:
        new_image[(dim - image.shape[0])//2:-(dim - image.shape[0])//2 ] = image
    return new_image
