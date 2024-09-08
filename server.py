
import torch
import numpy as np

from server import PromptServer
from aiohttp import web
import time

routes = PromptServer.instance.routes
@routes.post('/decode_latent')
async def my_function(request):
    the_data = await request.post()
    return web.json_response({"image":RemoteVaeServer.decode(the_data)})


class RemoteVaeServer:
    vae     = None
    called  = False

    category = "experimental"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."}),
            }}

    RETURN_TYPES = ()
    FUNCTION = "func"
    OUTPUT_NODE = True

    def func(self, vae):
        RemoteVaeServer.vae = vae
        self.called = False 
        while not self.called: time.sleep(10)
        return ()
    
    @classmethod
    def decode(cls, dict):
        latent_samples = torch.from_numpy( np.array( dict['latent' ]) )
        image:torch.Tensor = cls.vae.decode(latent_samples.cuda()).cpu()
        cls.called = True
        return image.numpy().tolist()