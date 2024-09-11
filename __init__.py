from .client import RemoteVae
from .server import RemoteVaeServer
from .save_promise import SavePromise

NODE_CLASS_MAPPINGS = {
    "Remote Vae" : RemoteVae,
    "Remote Vae Server" : RemoteVaeServer,
    "Save Later" : SavePromise,
}

__all__ = ['NODE_CLASS_MAPPINGS',]