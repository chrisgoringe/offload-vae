import requests
from tensor_rep import TensorRep
    
class RemoteVae:
    category = "experimental"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT",),
                    "server": ("STRING", {"default":"https://192.168.0.119:8188"}),
                }}

    def func(self, latent, server):
        payload = TensorRep.tensor_to_dict(latent['samples'])
        r       = requests.get(server+"/decode_latent", json=payload).json()
        image   = TensorRep.dict_to_tensor(r)
        return (image,)
