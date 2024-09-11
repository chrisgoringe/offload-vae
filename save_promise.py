from nodes import SaveImage
import asyncio, threading

class SavePromise(SaveImage):
    CATEGORY = "remote_offload"
    RETURN_TYPES = ()
    FUNCTION = "func"
    OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(s):
        it = SaveImage.INPUT_TYPES()
        required = {"promised_image": ("PROMISE", {"tooltip": "The output from Remote Vae set to mode='later'"})}
        for key in it['required']: 
            if key!='images': required[key] = it['required'][key]
        it['required'] = required
        return it

    def func(self, promised_image, **kwargs):
        def save_later():
            image = asyncio.run(promised_image)
            self.save_images(images=image, **kwargs)
        threading.Thread(target=save_later, daemon=True).start()
        return ()