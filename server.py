import asyncio
from websockets.asyncio.server import serve
import torch
from safetensors.torch import load_file
import numpy as np

async def main():
    async with serve(decode, "localhost", 8765, max_size=2**30):
        await asyncio.get_running_loop().create_future()  # run forever

class RemoteVaeServer:
    websocket = None
    category = "experimental"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."})
            }}

    RETURN_TYPES = ()
    FUNCTION = "func"
    OUTPUT_NODE = True

    def __init__(self):
        pass

    def func(self, vae):
        command = self.websocket.recv().decode()
        if command=="V":
            shape = self.websocket.recv()
            raw = self.websocket.recv()
            latent_samples = torch.from_numpy( np.ndarray(shape=shape, dtype=np.float32, buffer=raw) )
            image:torch.Tensor = vae.decode(latent_samples)
            self.websocket.send(image.shape)
            self.websocket.send(image.numpy().to_bytes())
        return ()
        

def decode(websocket):
    RemoteVaeServer.websocket = websocket
    
asyncio.run(main())