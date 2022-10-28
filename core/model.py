# Create model from smp - pytorch semantic segmentation models in pytorch

import segmentation_models_pytorch as smp

def create_model(output_channels: int = 1):
    """
    DeepLabv3 plus class with custom encoder and pretrained-weights
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the Unet model with the chosen encoder.
    """

    model = smp.DeepLabV3Plus(
              encoder_name="efficientnet-b2",       # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
              encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
              in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
              classes=output_channels,        # model output channels (number of classes in your dataset)
            )

    return model
