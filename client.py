import requests
import json
import tempfile
from .tensor_rep import TensorRep
    
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
        with tempfile.TemporaryFile(mode='+a') as fp:
            fp.write(TensorRep.tensor_to_str(latent['samples']))
            fp.seek(0)
            files = {'file': fp}
            r       = requests.post(server+"/decode_latent", files=files)
        image   = TensorRep.dict_to_tensor(r.json()['image'])
        return (image,)
