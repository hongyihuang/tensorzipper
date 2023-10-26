TensorZipper is a minimal neural net inference framework with built in compression management. It has a Python compiler based on PyTorch that exports a compressed bitstream file and a C/C++ runtime that can be ported to various platforms. 

This open source version intends to be a general starting point to demonstrate the capabilities of a neural network codec. 

To run the demo, open `/runtime/esp32-s3/cifar-10-fc` folder in arduino-ide and upload it to esp32 chips.

Train the model in `cifar-10-fully-connected.py`, load weights & biases using `cifar-10-load.py` to `runtime/esp32-s3/cifar-10-fc/cifar-10-fc.h` file to be consumed by C runtime, and evaluate the compute results in `cifar-10-eval.py` to check compute accuracy.

We are currently fixing some bugs to improve benchmark performance, please stay tuned. 