import torch, pickle
from websockets.sync.client import connect
import numpy as np

def decode(t:torch.Tensor, server):
    with connect(server, max_size=2**30) as websocket:
        websocket.send("V".encode())
        websocket.send(t.shape)
        websocket.send(t.numpy().astype(np.float32).to_bytes())

        response = websocket.recv()
        return pickle.loads(response)
    
class RemoteVae:
    category = "experimental"
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT",),
                    "server": ("STRING", {"default":"ws://localhost:8765"}),
                }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"

    def func(self, latent, server):
        image = decode(torch.Tensor(latent['samples']), server)
        return (image,)
