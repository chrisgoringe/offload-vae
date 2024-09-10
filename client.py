import requests
import tempfile
from .transformations import save_tensor_in_file, bytes_to_tensor
    
class RemoteVae:
    CATEGORY = "remote_offload"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"
    OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT",),
                    "server": ("STRING", {"default":"https://127.0.0.1:8188"}),
                    "wait": (["yes","no"], {})
                }}

    def func(self, latent, server, wait):
        with tempfile.TemporaryFile(mode='b+a') as fp:
            save_tensor_in_file(latent['samples'], fp)
            fp.seek(0)
            if wait=="yes":
                r = requests.post(server+"/decode_latent", files={'file': fp})
                image = bytes_to_tensor(r.content)
                return (image,)
            else:
                r = requests.post(server+"/decode_latent_noreply", files={'file': fp})
                return (None,)