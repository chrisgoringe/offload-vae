import requests
import tempfile
from .shared import save_tensor_in_file, bytes_to_tensor, latent_route_name, no_reply
from nodes import SaveImage
import asyncio, threading

def check_ok(r:requests.Response):
    if r.status_code!=200: raise Exception(f"Call to {r.url} failed: {r.reason}")

class SendLatent:
    CATEGORY = "remote_offload"
    RETURN_TYPES = ("ASYNC_IMAGE",)
    FUNCTION = "func"
    OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT",),
                    "server": ("STRING", {"default":"https://127.0.0.1:8188"}),
                    "forget": ("BOOLEAN", {"default":False, "tooltip":"If true, the return is None"})
                }}

    def func(self, latent, server, forget):
        async def get_reponse(l, forget=False):
            with tempfile.TemporaryFile(mode='b+a') as fp:
                save_tensor_in_file(l['samples'], fp)
                fp.seek(0)
                r = requests.post(server+latent_route_name+(no_reply if forget else ""), files={'file': fp})
                check_ok(r)
                return bytes_to_tensor(r.content) if not forget else None
        if forget:
            asyncio.run(get_reponse(latent, forget=True))
            return (None,)
        else:
            return (get_reponse(latent),)  
        
class SaveAsyncImage(SaveImage):
    CATEGORY = "remote_offload"
    RETURN_TYPES = ()
    FUNCTION = "func"
    OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(s):
        it = SaveImage.INPUT_TYPES()
        required = {"async_image": ("ASYNC_IMAGE", {"tooltip": "The output from LatentClient"})}
        for key in it['required']: 
            if key!='images': required[key] = it['required'][key]
        it['required'] = required
        return it

    def func(self, promised_image, **kwargs):
        def save_later():
            image = asyncio.run(promised_image)
            self.save_images(images=image, **kwargs)
        threading.Thread(target=save_later, daemon=True).start()
        return ()