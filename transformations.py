import torch
from safetensors.torch import save, load

def tensor_to_bytes(tensor:torch.Tensor) -> bytes:
    return save({"tensor":tensor.contiguous()})

def bytes_to_tensor(data:bytes) -> torch.Tensor:
    return load(data)['tensor']

def save_tensor_in_file(tensor:torch.Tensor, file_handle):
    file_handle.write(tensor_to_bytes(tensor))

def load_tensor_from_file(file_handle) -> torch.Tensor:
    return bytes_to_tensor (file_handle.read())

