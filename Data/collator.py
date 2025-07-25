import torch
from torchvision.transforms import functional as TF

class DataCollator:
    def __init__(self, pad = 0):
        self.pad = pad
    def __call__(self, data):

        # Find the maximum height and width
        max_height = max(item['images'].shape[1] for item in data)
        max_width = max(item['images'].shape[2] for item in data)

        # Find the max height and width in the batch of image for blood vessel segmentation
        max_height_image_bv = max(item['images_bv'].shape[1] for item in data)
        max_width_image_bv = max(item['images_bv'].shape[2] for item in data)

        padded_images = [] #Input image
        padded_images_odin = [] #Input optic disc, cup and lesion segmenter
        padded_images_bv = [] #input bv segmenter

        for item in data:
            # Calculate padding sizes
            padding_right = max_width - item['images'].shape[2]
            padding_bottom = max_height - item['images'].shape[1]

            padding_right_lunet = max_width_image_bv - item['images_bv'].shape[2]
            padding_bottom_lunet = max_height_image_bv - item['images_bv'].shape[1]



            # Pad the images and labels. Padding is applied only to bottom and right.
            padded_image = TF.pad(item['images'], padding=[0, 0, padding_right, padding_bottom], fill=0)
            padded_image_odin = TF.pad(item['images_odin'], padding=[0, 0, padding_right, padding_bottom], fill=0)
            padded_image_bv = TF.pad(item['images_bv'], padding=[0, 0, padding_right_lunet, padding_bottom_lunet], fill=0)


            padded_images.append(padded_image)
            padded_images_odin.append(padded_image_odin)
            padded_images_bv.append(padded_image_bv)


        output = {}
        output["images"] = torch.stack(padded_images, dim=0)
        output["images_odin"] = torch.stack(padded_images_odin, dim=0)
        output["images_bv"] = torch.stack(padded_images_bv, dim=0)

        return output
