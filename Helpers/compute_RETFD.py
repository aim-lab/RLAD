from glob import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision.transforms import transforms
from scipy.linalg import sqrtm
from tqdm import tqdm
from PIL import Image
from Models.RETFound import vit_large_patch16, interpolate_pos_embed, extrapolate_pos_emb

def load_retfound(model_pretrained="trained_checkpoints/RETFound/RETFound_cfp_weights.pth"):
    vit = vit_large_patch16(num_classes=2, drop_path_rate=0.2, global_pool=True, dynamic_img_size=True)
    checkpoint = torch.load(model_pretrained,weights_only=False, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = vit.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(vit, checkpoint_model)
    # load pre-trained model
    msg = vit.load_state_dict(checkpoint_model, strict=False)

    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    print("Loaded RetinaFound Successfully")
    return vit.cuda()
def pad_to_square(image, background_color=(0, 0, 0)):
    """
    Pads a rectangular image to make it square by adding padding to the shorter side.

    Args:
        image (PIL.Image.Image): Input image.
        background_color (tuple): Background color for padding (default: black).

    Returns:
        PIL.Image.Image: Square-padded image.
    """
    width, height = image.size
    if width == height:
        return image  # Already square

    # Determine the size of the square and calculate padding
    max_side = max(width, height)
    new_image = Image.new(image.mode, (max_side, max_side), background_color)

    # Paste the original image at the center of the new square canvas
    if width > height:
        new_image.paste(image, (0, (max_side - height) // 2))
    else:
        new_image.paste(image, ((max_side - width) // 2, 0))

    return new_image


def calculate_fid_from_paths(real_path_patterns, fake_path_pattern, backbone="inception",  batch_size=32):
    """
    Calculate the Fréchet Inception Distance (FID) between images matching multiple real path patterns and one fake path pattern.

    Args:
        real_path_patterns (list): List of glob patterns for real images.
        fake_path_pattern (str): Glob pattern for fake images.
        batch_size (int): Number of images to process in each batch.

    Returns:
        float: FID score.
    """
    # Load pre-trained Inception-v3 model
    if backbone == "inception":
        model = inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        model.fc = nn.Identity()  # Remove the final classification layer
        input_size = (299,299)
    else:
        model = load_retfound()
        input_size = (224,224)


    model.eval()

    # Move model to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def extract_features(path_patterns, model, batch_size, backbone = "inception"):
        """
        Extract features from images matching given glob patterns.

        Args:
            path_patterns (list): List of glob patterns for images.
            model: The pre-trained model.
            batch_size (int): Number of images to process in a batch.

        Returns:
            np.ndarray: Extracted features.
        """
        image_paths = []
        for pattern in path_patterns:
            image_paths.extend(glob(pattern, recursive=True))

        if not image_paths:
            raise ValueError("No images found for the provided patterns.")

        features = []
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size)):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                for path in batch_paths:
                    try:
                        image = Image.open(path).convert("RGB")  # Ensure 3-channel RGB
                        image = pad_to_square(image)
                        batch_images.append(preprocess(image))
                    except Exception as e:
                        print(f"Skipping file {path}: {e}")

                if batch_images:
                    batch_tensor = torch.stack(batch_images).to(device)
                    if backbone == "inception":
                        batch_features = model(batch_tensor)
                    else:
                        batch_features = model.forward_features(batch_tensor)
                    features.append(batch_features.cpu().numpy())

        return np.concatenate(features, axis=0)

    fake_features = extract_features(fake_path_pattern, model, batch_size, backbone=backbone)

    if type(real_path_patterns[0]) == list:
        fidscores = []
        for real_pattern in real_path_patterns:
            real_features = extract_features(real_pattern, model, batch_size, backbone=backbone)


            # Calculate mean and covariance for real and fake features
            mu_real = np.mean(real_features, axis=0)
            sigma_real = np.cov(real_features, rowvar=False)
            mu_fake = np.mean(fake_features, axis=0)
            sigma_fake = np.cov(fake_features, rowvar=False)

            # Compute FID
            diff = mu_real - mu_fake
            covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)

            # Handle numerical issues
            if np.iscomplexobj(covmean):
                covmean = covmean.real

            fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
            fidscores.append(fid)
            print(fid)
        return fidscores
    else:
        real_features = extract_features(real_path_patterns, model, batch_size,backbone=backbone)

        # Calculate mean and covariance for real and fake features
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)

        # Compute FID
        diff = mu_real - mu_fake
        covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)

        # Handle numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)

        return fid
