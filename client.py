import torch, pickle
from websockets.sync.client import connect

def decode(t:torch.Tensor, server):
    with connect(server) as websocket:
        b = pickle.dumps(t)
        websocket.send(b)
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
