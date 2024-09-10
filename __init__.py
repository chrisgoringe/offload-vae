from .client import RemoteVae
from .server import RemoteVaeServer

NODE_CLASS_MAPPINGS = {
    "Remote Vae" : RemoteVae,
    "Remote Vae Server" : RemoteVaeServer,
}

__all__ = ['NODE_CLASS_MAPPINGS',]