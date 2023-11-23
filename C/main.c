#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>

#include "ans.h"
#include "nanotorch.h"
#if !defined(ZIPLUT3) && !defined(ZIPLUT4)
#include "cifar-10-fc-runtime.h"
#endif

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
void stats(tzip *file) {
  size_t bin[16] = {0};
  uint16_t max = 0;
  for (size_t i = 257; i < 65536; i++) {
    size_t curr = i;
    size_t count = 0;

    while (curr > 256) {
      count++;
      curr = file->next_state[curr];
      if ((count == 5) && (curr > max)) max = curr;
    }
    bin[count]++;
  }
  printf("Max: %zu\n", max);
  printf("Bin counts: %zu %zu %zu %zu %zu %zu %zu\n", bin[0], bin[1], bin[2], bin[3], bin[4], bin[5], bin[6]);
}
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

void generateWeights(tzip *file, uint8_t *data, size_t size) {
  uint16_t lfsr = 0xACE1u;
  
  for (int i = 0; i < size; i++) {
    lfsr ^= lfsr >> 7;
    lfsr ^= lfsr << 9;
    lfsr ^= lfsr >> 13;

    lfsr += lfsr & 0xFF;

    // according to symbol distribution
    data[i] = file->inv_f[lfsr & 0xFF];
  }
}

const static uint8_t dist[16] = {1, 1, 2, 4, 10, 22, 55, 69, 54, 23, 9, 3, 1, 1, 1};
const static size_t RUNS = 32;
const static size_t SIZE = 1024*2048;
const static size_t TOTAL_SIZE = RUNS*SIZE;

void memPerf2() {
  tzip file;
  uint8_t *data = malloc(TOTAL_SIZE);
  int8_t *buffer = malloc(SIZE);
  int8_t *otherBuffer = malloc(SIZE);
  if (data == NULL) {
    printf("malloc data failed!");
    exit(0);
  }

  printf("Generating %zu Bytes of data\n", TOTAL_SIZE);
  
  init_tzip(&file, data, dist, TOTAL_SIZE); //64MB (32x2MB)
  file.rows = SIZE;

  printf("Generate Weights\n");
  generateWeights(&file, data, TOTAL_SIZE);
  printf("Compress\n");
  compress(&file, data, TOTAL_SIZE);

  //printf("Stats\n");
  //stats(&file);

  printf("File size = %zu\n", file.size);
  size_t err = 0;
  for (size_t i = 0; i < RUNS; i++) {
    beginms(INF_TIME_ID);
    memcpy(otherBuffer, data+i*SIZE, SIZE);
    endms(INF_TIME_ID);

    //printf("%hu, %zu\n", file.ckpt_state[0], file.ckpt_offset[0]);
    beginms(ZIP_TIME_ID);
    unzip(&file, buffer, SIZE);
    endms(ZIP_TIME_ID);
    //printf("%hu, %zu\n", file.ckpt_state[0], file.ckpt_offset[0]);

    for (size_t j = 0; j < SIZE; j++) {
      if ((buffer[j] + 8) != otherBuffer[j]) {
        //printf("%zu, %zu\n", i, j);
        //printf("%u != %u\n", (buffer[j] + 8), data[i*SIZE + j]);
        //sleep(1);
        //exit(0);
        err++;
      }
    }
  }
  printf("err = %zu\n", err);
  printms(INF_TIME_ID);
  printms(ZIP_TIME_ID);
}

int main() {
  printf("Welcome to TensorZipper!\n");

  // 1. Make a static memory on stack (SRAM)
  // memcpy(dest, src, size);
  // 2. Copy each row over from (flash)
  // 3. Multiply each row to the previous data
  // Inference
  printf("memPerf...\n"); 
  memPerf();

  printf("memPerf2...\n");  
  memPerf2();

  /* Disabled due to support only in ZIPLUT2 or below
   * In order to run at ZIPLUT3, need to modify python compiler
  */
  #if !defined(ZIPLUT3) && !defined(ZIPLUT4)
  cifar_setup();
  check_weights();
 
  printf("TensorZipper:\n");
  for (int i = 0; i < 64; i++) cifar_compressed();
  printms(READ_TIME_ID);
  printms(PROD_TIME_ID);

  printf("Baseline:\n");
  for (int i = 0; i < 64; i++) cifar_normal();
  printms(READ_TIME_ID);
  printms(PROD_TIME_ID);
  #endif

  return 0;
}

