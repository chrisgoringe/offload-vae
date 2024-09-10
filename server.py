import torch
from server import PromptServer
from aiohttp import web
from .transformations import load_tensor_from_file, tensor_to_bytes

routes = PromptServer.instance.routes
@routes.post('/decode_latent')
async def my_function(request):
    the_data = await request.post()
    if RemoteVaeServer.vae_laoded is not None:
        return web.Response(body = decode(the_data))
    else:
        return web.HTTPServerError(text="Server not running")

def decode(dict) -> bytes:
    f = dict['file']
    latent_samples = load_tensor_from_file(f.file)
    with torch.no_grad():
        image:torch.Tensor = RemoteVaeServer.vae_laoded.decode(latent_samples.cuda()).cpu()
    return tensor_to_bytes(image)

class RemoteVaeServer:
    vae_laoded = None

    CATEGORY = "remote_offload"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."}),
                "mode": (["start", "stop"],)
            }}

    RETURN_TYPES = ()
    FUNCTION = "func"

    def func(self, vae, mode):
        RemoteVaeServer.vae_laoded = vae if mode=="start" else None
        return ()
    
