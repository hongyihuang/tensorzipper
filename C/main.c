#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <sys/time.h>

#include "ans.h"
#include "nanotorch.h"
#include "cifar-10-fc-runtime.h"

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

void memPerf() {
  volatile int8_t *test = (int8_t*)malloc(1024*1024*16);
  volatile int32_t *test32 = (int32_t*)test;
  
  if (test == NULL) {
    PRINT("memPerf failed to malloc");
    return;
  }
  int64_t content = 0;

  printf("32 Million Params in Int4:\n");
  beginms(TWRITE_TIME_ID);
  for (int i = 0; i<1024*1024*16; i++)
    *(test+i) = i;

  endms(TWRITE_TIME_ID);
  printms(TWRITE_TIME_ID);

  printf("32 Million Params in Int16:\n");
  beginms(TWRITE_TIME_ID);
  for (int j = 0; j < 4*4; j++) {
    for (int i = 0; i<1024*1024*16/4; i++) {
      *(test32+i) = i*j;
    }
  }
  endms(TWRITE_TIME_ID);
  printms(TWRITE_TIME_ID);
  /*
  sprintf(buf, "Cycles per loop: %f\n", secs*3e9/(1024*1024*16));
  PRINT(buf);
  sprintf(buf, "Write MBps: %f\n", 16/secs);
  PRINT(buf);
  */
  printf("32 Million Params in Int4:\n");
  beginms(TREAD_TIME_ID);
  for (int i = 0; i<1024*1024*16; i++) {
    content += (*(test+i) * content);
  }
  endms(TREAD_TIME_ID);
  printms(TREAD_TIME_ID);

  printf("32 Million Params in Int16:\n");
  beginms(TREAD_TIME_ID);
  for (int j = 0; j < 4*4; j++) {
    for (int i = 0; i<1024*1024*16/4; i++) {
      content += (*(test32+i) * content);
    }
  }
  endms(TREAD_TIME_ID);
  printms(TREAD_TIME_ID);
  /*
  sprintf(buf, "Cycles per loop: %f\n", secs*3e9/(1024*1024*16.0));
  PRINT(buf);
  sprintf(buf, "Read MBps: %f\n", 16/secs);
  PRINT(buf);
  */

  sprintf(buf, "Result %lld \n", content);
  PRINT(buf);
  
  free(test);
}

int main() {
  printf("Welcome to TensorZipper!\n");

  // 1. Make a static memory on stack (SRAM)
  // memcpy(dest, src, size);
  // 2. Copy each row over from (flash)
  // 3. Multiply each row to the previous data
  // Inference
  memPerf();

  cifar_setup();
  check_weights();

  printf("TensorZipper:\n");
  for (int i = 0; i < 256; i++) cifar_compressed();
  printms(READ_TIME_ID);
  printms(PROD_TIME_ID);

  printf("Baseline:\n");
  for (int i = 0; i < 256; i++) cifar_normal();
  printms(READ_TIME_ID);
  printms(PROD_TIME_ID);
  
  return 0;
}

