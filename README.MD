![image](./logopng.png)

TensorZipper is a minimal neural net inference framework with built in compression management. It has a Python compiler based on PyTorch that exports a compressed bitstream file and a C/C++ runtime that can be ported to various platforms. 

This open source version intends to be a general starting point to demonstrate the capabilities of a neural network codec. 

Before doing anything, train the model in `cifar-10-fully-connected.py`, load weights & biases by running `cifar-10-load.py`. This generates a weight header file under `runtime/esp32-s3/cifar-10-fc/cifar-10-fc.h`. Rename the file as `cifar-10-fc-runtime.h` and copy it in both `/C`.

Two workflows: 
 1. PC: under the `/C` folder. Simply run `make` under that folder and run the executable `./main`. 
 2. Any arduino supported toolchains: Under the `runtime` folder. Copy the library in C folder, except `main.c, .gitignore, and Makefile`, open `/runtime/esp32-s3/cifar-10-fc/cifar-10-fc.ino` file in arduino-ide. To use embedded specific printf, timers, and vector optimization, go to `arch.h`, uncomment `#define ARDUINO`.

We are currently fixing some bugs to improve benchmark performance, please stay tuned. You will soon be able to run a specific layer in llama-2 and observe the 33% performance gain on any computer! 
