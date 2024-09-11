import torch
from server import PromptServer
from aiohttp import web
from .transformations import load_tensor_from_file, tensor_to_bytes
import time, threading, sys
from comfy.sd import VAE

class VaeMismatch(Exception):
    pass

routes = PromptServer.instance.routes
@routes.post('/decode_latent')
async def my_function(request):
    the_data = await request.post()
    if RemoteVaeServer.vae_loaded is not None:
        try:
            latent_samples = RemoteVaeServer.get_samples(the_data)
            return web.Response(body = RemoteVaeServer.decode(latent_samples))
        except VaeMismatch:
            return web.HTTPServerError(text=str(sys.exception()))
    else:
        return web.HTTPServerError(text="Server not running")
    
@routes.post('/decode_latent_noreply')
async def my_function(request):
    the_data = await request.post()
    if RemoteVaeServer.vae_loaded is not None:
        try:
            latent_samples = RemoteVaeServer.get_samples(the_data)
            threading.Thread(target=RemoteVaeServer.decode, args=(RemoteVaeServer,latent_samples,)).start()
        except VaeMismatch:
            return web.HTTPServerError(text=str(sys.exception()))
        return web.HTTPOk()
    else:
        return web.HTTPServerError(text="Server not running")

class RemoteVaeServer:
    vae_loaded:VAE              = None
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
    
    @classmethod
    def get_samples(cls, the_dict) -> torch.Tensor:
        f = the_dict['file']
        latent_samples = load_tensor_from_file(f.file)
        B, C, W, H = latent_samples.shape
        if cls.vae_loaded.latent_channels != C:
            raise VaeMismatch(f"Server has a {cls.vae_loaded.latent_channels} channel VAE loaded, a {C} channel latent was sent.")
        return latent_samples

    @classmethod
    def decode(cls,latent_samples) -> bytes:
        with torch.no_grad():
            cls.image_returned = cls.vae_loaded.decode(latent_samples.cuda()).cpu()
        return tensor_to_bytes(cls.image_returned)
        
