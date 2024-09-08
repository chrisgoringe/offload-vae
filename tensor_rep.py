import torch
import numpy as np

class TensorRep:
    @classmethod
    def tensor_to_dict(cls, tensor:torch.Tensor) -> dict:
        return { 
            "shape":list(x for x in tensor.shape),
            "dtype":f"{tensor.dtype}",
            "data" :tensor.cpu().numpy().tolist()
        }

    @classmethod
    def dict_to_tensor(cls, dict:dict) -> torch.Tensor:
        return torch.from_numpy(np.array(dict['data'])).to( getattr(torch, dict['dtype']) )
    