#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <sys/time.h>

#include "ans.h"
#include "nanotorch.h"
#include "cifar-10-fc-runtime.h"

//#include "ans.h"

/*
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

//uint8_t data[196], hidden_act[64], w_buf[196], out[10];

/*
Estimated memory footprint:
196*8 = 1568B
196 = 
*/

double elapsed(struct timeval begin, struct timeval end) {
  long seconds = end.tv_sec - begin.tv_sec;
  long microseconds = end.tv_usec - begin.tv_usec;
  return seconds + microseconds*1e-6;
}

void memPerf() {
  int8_t *test = (int8_t*)malloc(1024*1024*16);
  char buf[128];
  if (test == NULL) {
    PRINT("memPerf failed to malloc");
    return;
  }
  int64_t content = 0;
  struct timeval begin, end;
  double secs;

  gettimeofday(&begin, 0);
  for (int i = 0; i<1024*1024*16; i++) {
    *(test+i) = i;
  }
  gettimeofday(&end, 0);
  secs = elapsed(begin, end);
  sprintf(buf, "Write Time: %.3f seconds.\n", secs);
  PRINT(buf);
  sprintf(buf, "Cycles per loop: %f\n", secs*3e9/(1024*1024*16));
  PRINT(buf);
  sprintf(buf, "Write MBps: %f\n", 16/secs);
  PRINT(buf);
  
  gettimeofday(&begin, 0);
  for (int i = 0; i<1024*1024*16; i++) {
    content += (*((volatile int8_t*)test+i) * content);
  }
  gettimeofday(&end, 0);
  secs = elapsed(begin, end);
  sprintf(buf, "Read Time: %.3f seconds.\n", secs);
  PRINT(buf);
  sprintf(buf, "Cycles per loop: %f\n", secs*3e9/(1024*1024*16.0));
  PRINT(buf);
  sprintf(buf, "Write MBps: %f\n", 16/secs);
  PRINT(buf);

  sprintf(buf, "Result %lld \n", content);
  PRINT(buf);

  free(test);
}

int main() {
  printf("Welcome to TensorZipper, profiling memory bandwidth.\n");

  // 1. Make a static memory on stack (SRAM)
  // memcpy(dest, src, size);
  // 2. Copy each row over from (flash)
  // 3. Multiply each row to the previous data
  // Inference
  memPerf();

  cifar_setup();
  check_weights();
  cifar_compressed();
  cifar_normal();

  return 0;
}

