TensorZipper is a minimal neural net inference framework with built in compression management. It has a Python compiler based on PyTorch that exports a compressed bitstream file and a C/C++ runtime that can be ported to various platforms. 

This open source version intends to be a general starting point to demonstrate the capabilities of a neural network codec. 

[TODO] 0.0.1
Basic OP Codes (Maximum of 32)
[TODO]

Compression Modes (4)
1. Dense
2. Huffman Code
3. Asymmetrical Numerical System
4. Custom Extension (use string to match)

Quantization Modes (4)
1. int8
2. fp8
3. bf16
4. Custom Extension (use string to match)