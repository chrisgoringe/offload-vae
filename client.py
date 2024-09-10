import requests
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
            TensorRep.save_tensor_in_file(latent['samples'], fp)
            fp.seek(0)
            r = requests.post(server+"/decode_latent", files={'file': fp})
        #image = TensorRep.dict_to_tensor(r.json()['image'])
        image = TensorRep.from_bytes(r.content)
        return (image,)
