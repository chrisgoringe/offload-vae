import requests
import tempfile
from .transformations import save_tensor_in_file, bytes_to_tensor

def check_ok(r:requests.Response):
    if r.status_code!=200: raise Exception(f"Call to {r.url} failed: {r.reason}")

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
                    "mode": (["wait","forget","promise"], {})
                }}

    def func(self, latent, server, mode):
        if mode=="promise":
            async def get_image(l):
                with tempfile.TemporaryFile(mode='b+a') as fp:
                    save_tensor_in_file(l['samples'], fp)
                    fp.seek(0)
                    r = requests.post(server+"/decode_latent", files={'file': fp})
                    check_ok(r)
                    return bytes_to_tensor(r.content)
            return (None,get_image(latent),)  
        else:
            with tempfile.TemporaryFile(mode='b+a') as fp:
                save_tensor_in_file(latent['samples'], fp)
                fp.seek(0)
                if mode=="wait":
                    r = requests.post(server+"/decode_latent", files={'file': fp})
                    check_ok(r)
                    image = bytes_to_tensor(r.content)
                    return (image,None,)
                elif mode=="forget":
                    r = requests.post(server+"/decode_latent_noreply", files={'file': fp})
                    check_ok(r)
                    return (None,None,)
  
