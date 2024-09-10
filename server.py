import torch
from server import PromptServer
from aiohttp import web
from .transformations import load_tensor_from_file, tensor_to_bytes
import time, threading

routes = PromptServer.instance.routes
@routes.post('/decode_latent')
async def my_function(request):
    the_data = await request.post()
    if RemoteVaeServer.vae_loaded is not None:
        return web.Response(body = decode(the_data))
    else:
        return web.HTTPServerError(text="Server not running")
    
@routes.post('/decode_latent_noreply')
async def my_function(request):
    the_data = await request.post()
    if RemoteVaeServer.vae_loaded is not None:
        threading.Thread(target=decode, args=(the_data,)).start()
        return web.HTTPOk()
    else:
        return web.HTTPServerError(text="Server not running")

def decode(dict) -> bytes:
    f = dict['file']
    latent_samples = load_tensor_from_file(f.file)
    with torch.no_grad():
        RemoteVaeServer.image_returned = RemoteVaeServer.vae_loaded.decode(latent_samples.cuda()).cpu()
    return tensor_to_bytes(RemoteVaeServer.image_returned)



class RemoteVaeServer:
    vae_loaded                  = None
    image_returned:torch.Tensor = None

    CATEGORY = "remote_offload"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."}),
            }}
    
    @classmethod
    def IS_CHANGED(self, vae):
        return float("NaN")

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"
    OUTPUT_NODE = True

    def func(self, vae):
        RemoteVaeServer.vae_loaded     = vae
        RemoteVaeServer.image_returned = None
        while RemoteVaeServer.image_returned is None: time.sleep(5)
        return (RemoteVaeServer.image_returned, )
    
