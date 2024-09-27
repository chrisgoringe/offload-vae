import requests
import tempfile
from .shared import save_tensor_in_file, bytes_to_tensor, latent_route_name, no_reply
from nodes import SaveImage
import asyncio, threading, queue

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
                url = server+latent_route_name+(no_reply if forget else "")
                print(f"Post to {url }")
                r = requests.post(url, files={'file': fp}, verify=False)
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
        required = {"async_image": ("ASYNC_IMAGE", {"tooltip": "The output from LatentClient"}), "wait": ("BOOLEAN",{"default":False})}
        for key in it['required']: 
            if key!='images': required[key] = it['required'][key]
        it['required'] = required
        return it

    def func(self, async_image, wait, **kwargs):
        def save_later(q:queue.Queue):
            image = asyncio.run(async_image)
            result = self.save_images(images=image, **kwargs)
            if q: q.put(result)
        q = queue.SimpleQueue() if wait else None
        threading.Thread(target=save_later, args=[q,], daemon=True).start()
        return q.get() if wait else {}

    