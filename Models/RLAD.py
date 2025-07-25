from torch import nn
from Helpers.utils import load_yaml
import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_snr
from diffusers import DiTPipeline
from tqdm import tqdm
from .CustomDiTPipeline.transformer_2d import Transformer2DModel
from .SwinV2 import Swinv2Segmenter
from safetensors.torch import load_model, save_model
import copy
class Resize(nn.Module):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size

    def forward(self, x):
        # Resize the input tensor to the specified size
        return F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)

class FundusDiT(nn.Module):
    def print_param_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
    def __init__(self,model_args):
        super().__init__()
        self.config_ = load_yaml(model_args.model_config_path)

        self.model_pretrained = self.config_.get("model_pretrained", "facebook/DiT-XL-2-512")
        self.pipeline = DiTPipeline.from_pretrained(self.model_pretrained, torch_dtype=torch.bfloat16)
        self.vae = self.pipeline.vae
        self.noise_scheduler = self.pipeline.scheduler

        # Create an instance of our custom transformer
        config = self.pipeline.transformer.config
        self.transformer = Transformer2DModel.from_config(config)
        # Load the weights from the original transformer pipeline
        self.transformer.load_state_dict(self.pipeline.transformer.state_dict())

        self.pipeline.transformer = self.transformer.bfloat16()

        self.input_perturbation = self.config_.get("input_perturbation", False)
        self.noise_offset = self.config_.get("noise_offset", None)
        self.prediction_type =  self.config_.get("prediction_type", None)
        self.snr_gamma = self.config_.get("snr_gamma",None)

        ### Load bv_segmenter, odin and lesion segmenter
        self.bv_segmenter = Swinv2Segmenter(model_args,kw = "bv")
        self.odin = Swinv2Segmenter(model_args, kw = "od")
        self.lesion = Swinv2Segmenter(model_args, kw = "lesion")

        if self.config_["bv"].get("seg_load_weights_from", None) is not None:
            load_model(self.bv_segmenter, self.config_["bv"].get("seg_load_weights_from", None))
        if self.config_["od"].get("seg_load_weights_from", None) is not None:
            load_model(self.odin, self.config_["od"].get("seg_load_weights_from", None))
        if self.config_["lesion"].get("seg_load_weights_from", None) is not None:
            load_model(self.lesion, self.config_["lesion"].get("seg_load_weights_from", None))


        print("Freezing Segmenters and vae")
        for param in self.bv_segmenter.parameters():
            param.requires_grad = False
        for param in self.odin.parameters():
            param.requires_grad = False
        for param in self.lesion.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False

        self.proj_bv_to_latent = nn.Conv2d(
                in_channels=2, out_channels=3, kernel_size=1
            )
        self.proj_odin_to_latent = nn.Conv2d(
            in_channels=2, out_channels=3, kernel_size=1
        )
        self.proj_lesion_to_latent = nn.Conv2d(
            in_channels=4, out_channels=3, kernel_size=1
        )

        ## Define UI token
        self.transformer.register_parameter("prompt_token", nn.Parameter(torch.zeros(1, 1, 1152).bfloat16()))
        self.register_parameter("prompt_token_yes_seg", nn.Parameter(torch.ones(1, 1, 1152).bfloat16(), requires_grad=False)) ##bv segmentation
        self.register_parameter("prompt_token_no_seg", nn.Parameter(torch.zeros(1, 1, 1152).bfloat16(), requires_grad=False)) ##bv segmentation
        self.register_parameter("prompt_token_yes_odin", nn.Parameter(torch.ones(1, 1, 1152).bfloat16(), requires_grad=False))
        self.register_parameter("prompt_token_no_odin", nn.Parameter(torch.zeros(1, 1, 1152).bfloat16(),requires_grad=False))
        self.register_parameter("prompt_token_yes_lesion", nn.Parameter(torch.ones(1, 1, 1152).bfloat16(), requires_grad=False))
        self.register_parameter("prompt_token_no_lesion", nn.Parameter(torch.zeros(1, 1, 1152).bfloat16(), requires_grad=False))

        ## Define different positional embeding of bv, cd and lesion
        self.pos_embed_segmentation = copy.deepcopy(self.transformer.pos_embed) ##bv
        self.pos_embed_odin = copy.deepcopy(self.transformer.pos_embed) ##cd
        self.pos_embed_lesion = copy.deepcopy(self.transformer.pos_embed) ##lesion

        ## Loading pretrained weight if training is resumed
        if self.config_.get("pretrained_weights_from", None) is not None:
            load_model(self, self.config_.get("pretrained_weights_from", None), strict=False)

        self.print_param_info()

    def generate_structural_latent(self, segmenter, projection, pos_embed, input_image,input_image_dummy, p = None, threshold = 0.0, resize = None):
        if p >= threshold:
            segmentations_latent = segmenter(input_image)["logits"][:, :projection.weight.shape[1]]
            use_positive_prompt_token = True
        else:
            segmentations_latent = segmenter(input_image_dummy)["logits"][:, :projection.weight.shape[1]]
            use_positive_prompt_token = False

        if resize is not None:
            segmentations_latent = Resize(size=(512, 512))(segmentations_latent)
        segmentations_latent = projection(segmentations_latent)
        segmentations_latent = (F.sigmoid(segmentations_latent) - 0.5) / 0.5
        segmentations_latent = self.vae.encode(segmentations_latent).latent_dist.sample()
        segmentations_latent = segmentations_latent * self.vae.config.scaling_factor
        segmentations_latent = pos_embed(segmentations_latent)

        return segmentations_latent, use_positive_prompt_token

    def forward(
        self,
        images= None,
        images_bv=None,
        images_odin = None,
        output_attentions= None,
        output_hidden_states = None,
        return_dict = None,
    ):
        ## casting image to bfloat16  and applying imageNet standardisation
        images = images.bfloat16()
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)

        # Create dummy image for bv segmentation (high res) and od/lesion segmentation (low res) for unconditional training
        images_bv_dummy = (images_bv * 0 - imagenet_mean) / imagenet_std
        images_odin_dummy = (images_odin * 0 - imagenet_mean) / imagenet_std

        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()

            ##BV
            p = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample()
            segmentations_latent, use_positive_prompt_token = self.generate_structural_latent(self.bv_segmenter,projection = self.proj_bv_to_latent,pos_embed = self.pos_embed_segmentation, input_image=images_bv, input_image_dummy=images_bv_dummy, p = p, threshold = 0.5, resize = (512,512))
            prompt_token_seg = self.prompt_token_yes_seg if use_positive_prompt_token else self.prompt_token_no_seg

            ##OD
            p = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample()
            odin_latent, use_positive_prompt_token = self.generate_structural_latent(self.odin,projection = self.proj_odin_to_latent, pos_embed=self.pos_embed_odin, input_image=images_odin, input_image_dummy=images_odin_dummy, p=p, threshold=0.5)
            prompt_token_odin = self.prompt_token_yes_odin if use_positive_prompt_token else self.prompt_token_no_odin

            ##Lesion
            p = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample()
            lesion_latent, use_positive_prompt_token = self.generate_structural_latent(self.lesion,projection = self.proj_lesion_to_latent, pos_embed=self.pos_embed_lesion, input_image=images_odin, input_image_dummy=images_odin_dummy, p=p, threshold=0.1)
            prompt_token_lesion = self.prompt_token_yes_lesion if use_positive_prompt_token else self.prompt_token_no_lesion


        latents = latents * self.vae.config.scaling_factor
        prompt_token_conditioning = torch.concat((prompt_token_lesion,prompt_token_odin,prompt_token_seg), dim = 1) ## UI Token
        segmentations_latent = segmentations_latent + odin_latent + lesion_latent ## Combined structural conditioning
        ## Adding eoc token
        prompt_token_conditioning = prompt_token_conditioning.expand((segmentations_latent.shape[0], -1, -1)).to(
            segmentations_latent.device)
        ## Final conditioning input
        segmentations_latent = torch.concat((segmentations_latent,prompt_token_conditioning), dim=1)

        ## Compute noisy latent
        noise = torch.randn_like(latents)
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        if self.input_perturbation:
            new_noise = noise + self.input_perturbation * torch.randn_like(noise)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if self.input_perturbation:
            noisy_latents = self.noise_scheduler.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        # encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
        class_labels = [0] * images.shape[0]
        class_labels = torch.tensor(class_labels, device=latents.device).reshape(-1)

        # Get the target for loss depending on the prediction type
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(prediction_type=self.prediction_type)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        latent_channels = self.transformer.config.in_channels
        model_pred = self.transformer(noisy_latents,
                                      timestep=timesteps,
                                      class_labels=class_labels,
                                      multimodal_segmentation=segmentations_latent,
                                      return_dict=False)[0]


        if self.transformer.config.out_channels // 2 == latent_channels:
            model_pred, model_pred_sigma = torch.split(model_pred, latent_channels, dim=1)

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                dim=1
            )[0]
            if self.noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return {"loss": loss, "model_pred": model_pred, "target": target}

    @torch.no_grad()
    def generate(
            self,
            num_gen=1,
            images_bv = None,
            images_lesion=None,
            images_odin=None,
            guidance_scale: float = 4.0,
            num_inference_steps: int = 50,
            generator = None,
            inputed_segmentation_latent=None,
            save_segmentation = None,
            # strict = True, ##If strict = False, we allow the model to not necessarily follow the conditioning for lesion and od
    ):

        if type(guidance_scale) == tuple:
            guidance_scale = generate_random_floats(guidance_scale[0], guidance_scale[1], (images_bv.shape[0], 1, 1, 1))
            guidance_scale = guidance_scale.to(images_bv.device).bfloat16()


        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images_bv.device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images_bv.device)
        ## Create dummy image for unconditional generation (high res for bv segmenter)
        uncond_dummy = torch.zeros_like(images_bv)
        images_bv_uncond = (uncond_dummy * 0 - imagenet_mean) / imagenet_std
        images_bv_uncond = images_bv_uncond.bfloat16()

        ## Create dummy image for unconditional generation (low res for od and lesion)
        uncond_dummy2 = torch.zeros((images_bv.shape[0],3,512,512)).to(images_bv_uncond.device)
        images_odin_uncond = (uncond_dummy2 * 0 - imagenet_mean) / imagenet_std
        images_odin_uncond = images_odin_uncond.bfloat16()

        ######Conditional#####
        prompt_token_seg = self.prompt_token_yes_seg
        prompt_token_odin = self.prompt_token_yes_odin
        prompt_token_lesion = self.prompt_token_yes_lesion

        if images_lesion is None:
            images_lesion = images_odin_uncond
            # if strict == False:
            #     prompt_token_lesion = self.prompt_token_no_lesion

        if images_odin is None:
            images_odin = images_odin_uncond
            # if strict == False:
            #     prompt_token_odin = self.prompt_token_no_odin

        with torch.no_grad():
            segmentations_latent, _ = self.generate_structural_latent(self.bv_segmenter,
                                                                      projection=self.proj_bv_to_latent,
                                                                      pos_embed=self.pos_embed_segmentation,
                                                                      input_image=images_bv,
                                                                      input_image_dummy=images_bv,
                                                                      p=1.0, threshold=0.5,
                                                                      resize=(512, 512))

            odin_latent, _ = self.generate_structural_latent(self.odin,
                                                             projection=self.proj_odin_to_latent,
                                                             pos_embed=self.pos_embed_odin,
                                                             input_image=images_odin,
                                                             input_image_dummy=images_odin,
                                                             p=1.0, threshold=0.5)

            lesion_latent, _ = self.generate_structural_latent(self.lesion,
                                                               projection=self.proj_lesion_to_latent,
                                                               pos_embed=self.pos_embed_lesion,
                                                               input_image=images_lesion,
                                                               input_image_dummy=images_lesion,
                                                               p=1.0, threshold=0.1)

            ## Combined structural latent conditioning
            segmentations_latent = segmentations_latent + odin_latent + lesion_latent

            # UI Token
            prompt_token_conditioning = torch.concat((prompt_token_lesion, prompt_token_odin, prompt_token_seg), dim=1)
            prompt_token_conditioning = prompt_token_conditioning.expand((segmentations_latent.shape[0], -1, -1)).to(
                segmentations_latent.device)
            ## Final conditioning input
            segmentations_latent = torch.concat((segmentations_latent, prompt_token_conditioning), dim=1)

            ### UNCONDITIONNAL
            segmentations_latent_uncond, _ = self.generate_structural_latent(self.bv_segmenter,
                                                                      projection=self.proj_bv_to_latent,
                                                                      pos_embed=self.pos_embed_segmentation,
                                                                      input_image=images_bv_uncond,
                                                                      input_image_dummy=images_bv_uncond,
                                                                      p=1.0, threshold=0.5,
                                                                      resize=(512, 512))

            odin_latent_uncond, _ = self.generate_structural_latent(self.odin,
                                                                                     projection=self.proj_odin_to_latent,
                                                                                     pos_embed=self.pos_embed_odin,
                                                                                     input_image=images_odin_uncond,
                                                                                     input_image_dummy=images_odin_uncond,
                                                                                     p=1.0, threshold=0.5)

            lesion_latent_uncond, _ = self.generate_structural_latent(self.lesion,
                                                                                       projection=self.proj_lesion_to_latent,
                                                                                       pos_embed=self.pos_embed_lesion,
                                                                                       input_image=images_odin_uncond,
                                                                                       input_image_dummy=images_odin_uncond,
                                                                                       p=1.0, threshold=0.1)


            ## Combined undond latent
            segmentations_latent_uncond = segmentations_latent_uncond + odin_latent_uncond + lesion_latent_uncond

            ## UI uncond token
            prompt_token_seg_uncond = self.prompt_token_no_seg
            prompt_token_odin_uncond = self.prompt_token_no_odin
            prompt_token_lesion_uncond = self.prompt_token_no_lesion
            prompt_token_conditioning_uncond = torch.concat((prompt_token_lesion_uncond, prompt_token_odin_uncond, prompt_token_seg_uncond), dim=1)
            prompt_token_conditioning_uncond = prompt_token_conditioning_uncond.expand((segmentations_latent.shape[0], -1, -1)).to(
                segmentations_latent.device)
            ## Final conditioning input (for unconditionnal dummy generation)
            segmentations_latent_uncond = torch.concat((segmentations_latent_uncond, prompt_token_conditioning_uncond), dim=1)

        ##### DDPM guided free
        class_labels = [0] * num_gen
        batch_size = len(class_labels)
        class_labels = torch.tensor(class_labels, device=self.transformer.device).reshape(-1)


        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        latents = self.randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self.transformer.device,
            dtype=self.transformer.dtype,
        )
        latent_model_input = torch.cat([latents] * 2) if guidance_scale is not None else latents
        segmentations_latent = torch.cat([segmentations_latent,segmentations_latent_uncond]) if guidance_scale is not None else segmentations_latent



        # class_null = torch.tensor([1000] * batch_size, device=self.transformer.device)
        class_labels_input = torch.cat([class_labels, class_labels], 0) if guidance_scale is not None else class_labels

        # set step values
        self.noise_scheduler.set_timesteps(num_inference_steps)
        for t in tqdm(self.noise_scheduler.timesteps):
            if guidance_scale is not None:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(timesteps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(latent_model_input.shape[0])
            # predict noise model_output
            with torch.no_grad():
                noise_pred = self.transformer(latent_model_input,
                                              timestep=timesteps,
                                              class_labels=class_labels_input,
                                              multimodal_segmentation=segmentations_latent,
                                              ).sample

            # perform guidance
            if guidance_scale is not None:
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.noise_scheduler.step(model_output, t, latent_model_input).prev_sample

        if guidance_scale is not None:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()



        return samples

    def randn_tensor(
            self,
            shape,
            generator = None,
            device = None,
            dtype = None,
            layout = None,
):
        """A Helpers function to create random tensors on the desired `device` with the desired `dtype`. When
        passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
        is always created on the CPU.
        """
        # device on which tensor is created defaults to device
        rand_device = device
        batch_size = shape[0]

        layout = layout or torch.strided
        device = device or torch.device("cpu")

        if generator is not None:
            gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
            if gen_device_type != device.type and gen_device_type == "cpu":
                rand_device = "cpu"
                if device != "mps":
                    print(
                        f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                        f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                        f" slighly speed up this function by passing a generator that was created on the {device} device."
                    )
            elif gen_device_type != device.type and gen_device_type == "cuda":
                raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

        # make sure generator list of length 1 is treated like a non-list
        if isinstance(generator, list) and len(generator) == 1:
            generator = generator[0]

        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
                for i in range(batch_size)
            ]
            latents = torch.cat(latents, dim=0).to(device)
        else:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

        return latents


def generate_random_floats(min_val, max_val, size=None):
    # Generate random floats in the range [0, 1]
    random_floats = torch.rand(size)
    # Scale and shift to the desired range [min_val, max_val]
    scaled_random_floats = min_val + (max_val - min_val) * random_floats
    return scaled_random_floats