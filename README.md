# Offload latent

A set of nodes to allow you to send a latent to another instance of Comfy (on the same, or a different, machine) to be processed and returned or saved.

There are two basic use cases: one where the server completes the workflow (presumably including saving the final image), and the other where the server
returns the image to be saved locally.

Basic usage is to set up the server with a workflow that start with a `Latent Server` node, and ends either with a `Save Image` or `Image Response`,
and then run the workflow with Auto Queue turned on. The `Latent Server` will wait to be called, then pass the latent to the rest of the workflow;
and when the workflow completes, it will restart and wait again.

## Server Completion

On the client side, use the `Send Latent` node, and set `forget` to `True`. When this node receives a latent it will send it to the server, and then exit
immediately (the output will be `None`, and should not be used)

On the server side, use the `Latent Server` node, ignore the `ResponseQueue` output, and just process the latent.

## Server Return

To get the image back, set `forget` to `False` and connect the `ASYNC_IMAGE` output to a `Save Async Image` node.

On the server side, process the latent, and then feed the resulting image and the `ResponseQueue` output to an `Image Response` node.

