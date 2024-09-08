import torch
import requests
import numpy as np
    
class RemoteVae:
    category = "experimental"
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT",),
                    "server": ("STRING", {"default":"https://192.168.0.119:8188"}),
                }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"

    def func(self, latent, server):
        payload = { "latent" : latent['samples'].numpy().toarray() }
        r = requests.get(server+"/decode_latent", json=payload)
        image = torch.from_numpy( np.array(r['image']) )
        return (image,)
