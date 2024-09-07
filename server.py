import asyncio
from websockets.asyncio.server import serve
from diffusers import AutoencoderKL
import torch
import pickle
import os
import json
from safetensors.torch import load_file

class Decoder:
    def __init__(self, vae_file, config):
        self.model:AutoencoderKL = AutoencoderKL(
            down_block_types = ("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D",),
            up_block_types   = ("UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D",),
            block_out_channels = (128,256,512,512),
            layers_per_block = 2,
            latent_channels = 16,
            sample_size = 512,
            scaling_factor = 0.3611,
            shift_factor = 0.1159,
            use_quant_conv = False,
            use_post_quant_conv = False,
        )
        sd = load_file(vae_file)
        self.model.load_state_dict(sd)
        self.model.cuda()


def path(*args): return os.path.join(os.path.dirname(__file__), *args)

with open(path("vae","vae_config.json"), 'r') as f: config = json.load(f)
decoder = Decoder(path("vae","diffusion_pytorch_model.safetensors"), config)

async def decode(websocket):
    message:bytes
    async for message in websocket:
        batch:torch.Tensor = pickle.loads(message)
        i = decoder.model.decode(batch.cuda(), return_dict=False)[0].cpu()
        await websocket.send(pickle.dumps(i))

async def main():
    async with serve(decode, "localhost", 8765):
        await asyncio.get_running_loop().create_future()  # run forever

asyncio.run(main())