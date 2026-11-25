

from .unet import UNet


def build_model(config):
    """Build the segmentation model specified by a saved experiment."""
    encoder = config.get("encoder", "scratch")
    if encoder == "scratch":
        return UNet(base_channels=config["base_channels"])
    raise ValueError(f"Encoder {encoder!r} has no registered U-Net implementation")
