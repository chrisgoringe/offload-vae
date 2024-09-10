import torch
import numpy as np
import json
from safetensors.torch import save, load

'''
A dict is used for the response, a filehandle for the send
'''

SAFETENSORS = True

class TensorRep:
    @classmethod
    def tensor_to_dict(cls, tensor:torch.Tensor) -> dict:
        return { 
            "shape":list(x for x in tensor.shape),
            "dtype":f"{tensor.dtype}".split(".")[-1],
            "data" :tensor.cpu().numpy().tolist()
        }

    @classmethod
    def dict_to_tensor(cls, dict:dict) -> torch.Tensor:
        return torch.from_numpy(np.array(dict['data'])).to( getattr(torch, dict['dtype']) )
    
    @classmethod
    def tensor_to_str(cls, tensor:torch.Tensor) -> str:
        return json.dumps(cls.tensor_to_dict(tensor))
    
    @classmethod
    def to_bytes(cls, tensor:torch.Tensor) -> bytes:
        return save({"tensor":tensor.contiguous()})
    
    @classmethod
    def from_bytes(cls, data:bytes) -> torch.Tensor:
        return load(data)['tensor']
    
    @classmethod
    def save_tensor_in_file(cls, tensor:torch.Tensor, file_handle):
        if SAFETENSORS:
            file_handle.write(cls.to_bytes(tensor))
        else:
            file_handle.write(TensorRep.tensor_to_str(tensor))

    @classmethod
    def load_tensor_from_file(cls, file_handle) -> torch.Tensor:
        if SAFETENSORS:
            return load(file_handle.read())['tensor']
        else:
            return TensorRep.dict_to_tensor(json.loads(file_handle.read()))
    
