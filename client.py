import requests
import tempfile
from .transformations import save_tensor_in_file, bytes_to_tensor
    
class RemoteVae:
    CATEGORY = "remote_offload"
    RETURN_TYPES = ("IMAGE","PROMISE",)
    FUNCTION = "func"
    OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT",),
                    "server": ("STRING", {"default":"https://127.0.0.1:8188"}),
                    "mode": (["wait","forget","later"], {})
                }}

    def func(self, latent, server, mode):
        with tempfile.TemporaryFile(mode='b+a') as fp:
            save_tensor_in_file(latent['samples'], fp)
            fp.seek(0)
            if mode=="wait":
                r = requests.post(server+"/decode_latent", files={'file': fp})
                image = bytes_to_tensor(r.content)
                return (image,None,)
            elif mode=="forget":
                r = requests.post(server+"/decode_latent_noreply", files={'file': fp})
                return (None,None,)
            elif mode=="async":
                async def get_image():
                    r = requests.post(server+"/decode_latent", files={'file': fp})
                    return bytes_to_tensor(r.content)
                return (None,get_image(),)     