from .client import SendLatent, SaveAsyncImage
from .server import LatentServer, ImageResponse

NODE_CLASS_MAPPINGS = {
    "Send Latent to Server" : SendLatent,
    "Save Async Image" : SaveAsyncImage,

    "Latent Server" : LatentServer,
    "Image Response" : ImageResponse,
}

__all__ = ['NODE_CLASS_MAPPINGS',]