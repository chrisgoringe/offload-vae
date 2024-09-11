# Offload VAE

A set of nodes to allow you to run the VAE decoder on another machine.

## Modes of operation

- *Wait* - the latent is sent to the server, and the client waits for it to be returned. The only benefit to this is if your are
close to VRAM limits (who isn't?), so saving the time offloading and loading models.

- *Forget* - the latent is sent to the server, and decode and saved there. The client can get on with the next run. Useful mostly for batch jobs

- *Later* - the latent is sent to the server, and the node returns a 'PROMISE'. The `Save Promised Image` node will save the image when it becomes available, but won't stop other nodes from running.



