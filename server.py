import torch
from server import PromptServer
from aiohttp import web
from .shared import load_tensor_from_file, tensor_to_bytes, latent_route_name, no_reply
import queue

class Server: pass

class NopQueue:
    def put(self, _): pass
    def get(self): return None 

class Dispatcher:
    '''
    Holds a collection of queues and allows jobs to be dispatched to them.
    There can only be one queue per __class__.

    A server calls:

    q = Dispatcher.ready(self.__class__)
    message, reply_queue = q.get()

    to indicate it is available, and to wait for a message (dict) with and reply_queue for the result

    Dispatcher.done(self.__class__) is used to indicate that the server is no longer available

    To send a message, use Dispatcher.dispatch_to_server, which returns a queue that the response will come on
    (NopQueue, which just swallows messages, is used if no_reply is set to True)

    '''
    server_queues:dict[type,queue.Queue] = {}
    @classmethod
    def dispatch_to_server(cls, clazz:type, message:dict, no_reply=False) -> queue.Queue:
        if (q := cls.server_queues.get(clazz.__name__, None)) is None:
            return None
        else:
            reply_q = NopQueue() if no_reply else queue.SimpleQueue()
            q.put((message, reply_q))
            return reply_q
    
    @classmethod
    def ready(cls, clazz:type) -> queue.Queue:
        cls.server_queues[clazz.__name__] = queue.SimpleQueue()
        return cls.server_queues[clazz.__name__]
    
    @classmethod
    def done(cls, clazz:type) -> None: 
        cls.server_queues.pop(clazz, None)


async def _dispatch(clazz, request, no_reply=False):
    '''
    Deal with a request that shiould be sent to the specified clazz of server
    '''
    the_data = await request.post()
    reply_queue:queue.Queue = Dispatcher.dispatch_to_server(clazz=clazz, message=the_data, no_reply=no_reply)
    if reply_queue:
        if no_reply: 
            return web.HTTPOk()
        else:
            response = reply_queue.get()
            return web.Response(body = response)
    else:
        return web.HTTPServerError(text="Server not running")

routes = PromptServer.instance.routes

'''
Create two routes for latent (/send_latent and /send_latent_noreply) which will go to the LatentServer instance
'''
@routes.post(latent_route_name)
async def dispatch(request, no_reply=False):
    return await _dispatch(LatentServer, request, no_reply)

@routes.post(latent_route_name+no_reply)
async def dispatch_noreply(request):
    return await dispatch(request, no_reply=True)
    
class LatentServer(Server):
    '''
    ComfyUI node that, when executed:
    - Registers to receive latents, then waits for one
    - Returns the latent and a RESPONSE_QUEUE
    - Deregisters and exits
    Normal use would be in a workflow with AutoQueue
    '''
    CATEGORY = "remote_offload"
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { }}
    
    @classmethod
    def IS_CHANGED(self, **kwargs): return float("NaN")

    RETURN_TYPES = ("LATENT", "RESPONSE_QUEUE",)
    FUNCTION = "func"

    def func(self):
        q = Dispatcher.ready(self.__class__)
        try:
            message, reply_queue = q.get()
            latent_samples = load_tensor_from_file(message['file'].file)
            return ( {'samples':latent_samples}, reply_queue )
        except:
            return ( None, None )
        finally:
            Dispatcher.done(self)
    
class ImageResponse:
    '''
    ComfyUI node that, when executed:
    - Sends the IMAGE it receives to the RESPONSE_QUEUE
    Normal use would be in a workflow with AutoQueue
    '''
    CATEGORY = "remote_offload"
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE", {}), "request": ("RESPONSE_QUEUE", {})}}   
    
    @classmethod
    def IS_CHANGED(self, **kwargs): return float("NaN")

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "func"

    def func(self, image:torch.Tensor, request:queue.Queue):
        request.put( tensor_to_bytes(image) )
        return ()

