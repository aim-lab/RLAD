from torch import nn
from Helpers.utils import load_yaml
from transformers import Swinv2Model,Swinv2Config
import math

class Swinv2Segmenter(nn.Module):
    def __init__(self,model_args, kw = None):
        super().__init__()
        self.config_ = load_yaml(model_args.model_config_path)
        if kw is not None:
            self.config_ = self.config_[kw]
        self.model_arch = "swin"
        # Load the configuration of the pre-trained model
        self.vit_patches_size = int(self.config_.get("vit_patches_size", 4))

        self.num_classes = int(self.config_.get("num_classes", 4))
        self.is_multilabel = self.config_.get("multilabel", False)

        self.model_pretrained = self.config_.get("seg_model_pretrained", "microsoft/swinv2-tiny-patch4-window8-256")
        print("config used:", self.config_.get("seg_model_pretrained"))

        config = Swinv2Config.from_pretrained(self.model_pretrained)
        config.image_size = 1444
        config.patch_size = self.vit_patches_size

        #image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.swinv2 = Swinv2Model.from_pretrained(self.model_pretrained, config = config)
        num_features = int(config.embed_dim * 2 ** (config.num_layers - 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features, out_channels=config.encoder_stride**2 * self.num_classes, kernel_size=1
            ),
            nn.PixelShuffle(config.encoder_stride),
        )


    def forward(
        self,
        images= None,
        segmentations = None,
        bool_masked_pos = None,
        head_mask = None,
        output_attentions= None,
        output_hidden_states = None,
        return_dict = None,
    ):

        outputs = self.swinv2(
            images,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output.transpose(1, 2)
        batch_size, num_channels, sequence_length = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        out = self.decoder(sequence_output)

        return {"logits": out}
