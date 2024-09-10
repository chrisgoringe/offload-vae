import torch
import numpy as np
import json

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