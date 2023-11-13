#include <stdlib.h>
#include "ans.h"
#include "nanotorch.h"
#include "arch.h"
#include "cifar-10-fc-param.h"

const static int16_t X_SIZE = 3*32*32; // 3072
const static int16_t H1_SIZE = sizeof(fc1_b); // 1000
const static int16_t H2_SIZE = sizeof(fc2_b); // 500
const static int16_t H3_SIZE = sizeof(fc3_b); // 10

#define neural_t int8_t

static neural_t *x, *w, *h1, *h2, *h3;
static tzip *file, *file2;
static uint32_t counter = 0;

void cifar_setup() {  
  x = (neural_t*)malloc(X_SIZE * sizeof(neural_t));
  w = (neural_t*)malloc(X_SIZE * sizeof(neural_t));
  h1 = (neural_t*)malloc(H1_SIZE * sizeof(neural_t));
  h2 = (neural_t*)malloc(H2_SIZE * sizeof(neural_t));
  h3 = (neural_t*)malloc(H3_SIZE * sizeof(neural_t));
  file = (tzip*)malloc(sizeof(tzip));
  file2 = (tzip*)malloc(sizeof(tzip));
  if (x == NULL || w == NULL || h1 == NULL || h2 == NULL || h3 == NULL || file == NULL || file2 == NULL) {
    while (1) {
      PRINT("malloc failed! \n");
      #ifdef ARDUINO
      delay(1000);
      #else
      exit(-1);
      #endif
    }
  }
  for (int i = 0; i < X_SIZE; i++) {
    x[i] = img[i];
  }
  
  sprintf(buf, "DIMS: %hd x %hd x %hd x %hd", X_SIZE, H1_SIZE, H2_SIZE, H3_SIZE);
}


void cifar_normal() {
  //PRINT("Layer 1: \n");
  file->data = fc1_w;
  file->size = sizeof(fc1_w);
  file->ckpt_offset = 0;
  fcRELU(file, M1, w, fc1_b, x, h1, X_SIZE, H1_SIZE, S2, inflateInt4_to_8);
  //PRINT("Results: ");
  //printArr(h1, H1_SIZE, 18);

  //PRINT("Layer 2: \n");
  file->data = fc2_w;
  file->size = sizeof(fc2_w);
  file->ckpt_offset = 0;
  fcRELU(file, M2, w, fc2_b, h1, h2, H1_SIZE, H2_SIZE, S3, inflateInt4_to_8);
  //PRINT("Results: ");
  //printArr(h2, H2_SIZE, 18);

  //PRINT("Layer 3: \n");
  file->data = fc3_w;
  file->ckpt_offset = 0;
  fc(file, M3, w, fc3_b, h2, h3, H2_SIZE, H3_SIZE, inflateInt4_to_8);
  //PRINT("Results: \n");
  //printArr(h3, H3_SIZE, 18);
}


void cifar_compressed() {
  //PRINT("Layer 1: \n");
  init_tzip(file, fc1_w_d, fc1_w_f, sizeof(fc1_w_d));
  fcRELU(file, M1, w, fc1_b, x, h1, X_SIZE, H1_SIZE, S2, unzip);
  //PRINT("Results: ");
  //printArr(h1, H1_SIZE, 18);

  //PRINT("Layer 2: \n");
  init_tzip(file, fc2_w_d, fc2_w_f, sizeof(fc2_w_d));
  fcRELU(file, M2, w, fc2_b, h1, h2, H1_SIZE, H2_SIZE, S3, unzip);
  //PRINT("Results: ");
  //printArr(h2, H2_SIZE, 18);

  //PRINT("Layer 3: \n");
  init_tzip(file, fc3_w_d, fc3_w_f, sizeof(fc3_w_d));
  fc(file, M3, w, fc3_b, h2, h3, H2_SIZE, H3_SIZE, unzip);
  //PRINT("Results: \n");
  //printArr(h3, H3_SIZE, 18);
}


void check_weights() {
  int8_t *vector, *vector_zip;
  //PRINT("Layer 1: ");

  vector = (int8_t*)malloc(H1_SIZE * sizeof(int8_t));
  vector_zip = (int8_t*)malloc(H1_SIZE * sizeof(int8_t));

  if (vector == NULL || vector_zip == NULL) {
    while (1) {
      PRINT("MALLOC in check_weights failed!");
      #ifdef ARDUINO
      delay(5000);
      #endif
    }
  }

  uint32_t startCounter, endCounter, inflateCounter, zipCounter;
  
  init_tzip(file2, fc1_w_d, fc1_w_f, sizeof(fc1_w_d));

  file->data = fc1_w;
  file->size = sizeof(fc1_w);
  file->ckpt_offset = 0;
  
  // unzip H1_SIZE for X_SIZE times, compare it to uncompressed version
  for (int i = 0; i < X_SIZE; i++) {
    beginms(ZIP_TIME_ID);
    size_t unzipSize = unzip(file2, vector_zip, H1_SIZE);
    endms(ZIP_TIME_ID);

    beginms(INF_TIME_ID);
    size_t inflateSize = inflateInt4_to_8(file, vector, H1_SIZE);
    endms(INF_TIME_ID);

    for (size_t j = 0; j < H1_SIZE; j++) {
      if (vector[j] != vector_zip[j]) {
        sprintf(buf, "i %zu, j %zu: %d != %d \n", i, j, vector[j], vector_zip[j]);
        PRINT(buf);
        exit(-1);
      }
    }
  }
  /*
  Serial.println("Layer 2: ");
  init_tzip(&zipFile, fc2_w_d, fc2_w_f, dist, inv_f, cumulative, sizeof(fc2_w_d));

  file.data = fc2_w;
  file.inv_f = NULL;
  file.size = sizeof(fc2_w);
  file.ckpt_offset = 0;

  for (int i = 0; i < H1_SIZE; i++) {
    size_t inflateSize = inflateInt4(&file, buf, H2_SIZE);
    size_t unzipSize = unzip(&zipFile, zipBuf, H2_SIZE);

    for (int j = 0; j < H2_SIZE; j++) {
      if (zipBuf[j] != buf[j]) {
        Serial.print(j);
        Serial.print(": ");
        Serial.print(zipBuf[j]);
        Serial.print(" != ");
        Serial.println(buf[j]);
      }
    }
  }

  Serial.println("Layer 3: ");
  init_tzip(&zipFile, fc3_w_d, fc3_w_f, dist, inv_f, cumulative, sizeof(fc3_w_d));

  file.data = fc3_w;
  file.inv_f = NULL;
  file.size = sizeof(fc3_w);
  file.ckpt_offset = 0;

  for (int i = 0; i < H2_SIZE; i++) {
    size_t inflateSize = inflateInt4(&file, buf, H3_SIZE);
    size_t unzipSize = unzip(&zipFile, zipBuf, H3_SIZE);

    for (int j = 0; j < H3_SIZE; j++) {
      if (zipBuf[j] != buf[j]) {
        Serial.print(j);
        Serial.print(": ");
        Serial.print(zipBuf[j]);
        Serial.print(" != ");
        Serial.println(buf[j]);
      }
    }
  }
  */
  printms(ZIP_TIME_ID);
  printms(INF_TIME_ID);
  sprintf(buf, "file.size: %zu, zipFile.size: %zu \n", file->size, file2->size);
  PRINT(buf);

  free(vector);
  free(vector_zip);
}