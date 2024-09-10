import requests
import tempfile
from .transformations import save_tensor_in_file, bytes_to_tensor
    
class RemoteVae:
    CATEGORY = "remote_offload"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT",),
                    "server": ("STRING", {"default":"https://127.0.0.1:8188"}),
                }}

    def func(self, latent, server):
        with tempfile.TemporaryFile(mode='b+a') as fp:
            save_tensor_in_file(latent['samples'], fp)
            fp.seek(0)
            r = requests.post(server+"/decode_latent", files={'file': fp})
        image = bytes_to_tensor(r.content)
        return (image,)
