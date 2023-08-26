#include <stdio.h>
#include <cstdint>
#include <stdlib.h>
#include <stddef.h>

//#include "ans.h"

/*
Tasks
1. Put compression into ans.h and ans.cpp
2. Put Neural Network into lil-tensor (test in python then copy from BearlyML).
3. Export weights automatically into an h in python.

Pre-HotChips
1. Write python script of hand drawing mnist digits
2. Port this to bearlyml

Things to wait:
1. Refactor the python code and move the toolchain local.
2. Change makefile to https://bazel.build
*/


/*
Memory Layout
- SRAM: act (mult of 8), weight (only each row)
- Flash: code, data, weight

File Layout
- main: this file, user code
- nanotorch: quantized matmul + bias operation
- ans: compression
- mnist: data, weights, and helper func (ascii art and uart load)
*/

/*
MNIST Spec
Data 14*14 = 196
hidden_dim = 64
Data size = 32
*/

uint8_t data[196], hidden_act[64], w_buf[196], out[10];

/*
Estimated memory footprint:
196*8 = 1568B
196 = 
*/

int main() {
  printf("Welcome to TensorZipper, running a MNIST demo.");
  // 1. Make a static memory on stack (SRAM)
  // memcpy(dest, src, size);
  // 2. Copy each row over from (flash)
  // 3. Multiply each row to the previous data
  // Inference
  return 0;
}

