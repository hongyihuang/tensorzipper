#include "esp_dsp.h"
#include "./cifar-10-fc.h"

typedef struct {
  int16_t* data;
  size_t size;
} array;

typedef struct {
  const uint8_t* data; // [size]
  const uint8_t *dist; // [16]
  uint8_t *inv_f, *cumulative; // inv_f is [256], cumulative is [16]
  size_t size;
  size_t ckpt_offset;
  uint32_t ckpt_state;
} tzip;

const static int16_t X_SIZE = 3*32*32; // 3072
const static int16_t H1_SIZE = sizeof(fc1_b); // 1000
const static int16_t H2_SIZE = sizeof(fc2_b); // 500
const static int16_t H3_SIZE = sizeof(fc3_b); // 10

static int16_t *x, *w, *h1, *h2, *h3;

/*
repeatedly call to unzip the file into buffer
Returns size of deflated buffer size
*/
// Currently is int16 to maintain compatibility with esp32_dotprod_s16
size_t unzip(tzip* file, int16_t* buf, size_t buf_size) {
  // initialize pointers and states
  // while loop decompress
  // filled buffer
  
  uint32_t state;
  uint8_t symbol, slot;
  // i: counter of decompressed buffer
  // j: counter of compressed bitstream
  size_t i = 0;
  size_t j;

  if (file->ckpt_offset) {
    // restore checkpoint
    state = file->ckpt_state;
    j = file->ckpt_offset;
  } else {
    // no checkpoint, initialize
    state = (file->data[0] << 16) + (file->data[1] << 8) + file->data[2];
    j = 3;
  }
  
  while (((j < file->size) || (state != 256)) && (i < buf_size)) {
    // perform decoding
    /*
    Serial.print("i, j, state: ");
    Serial.print(i);
    Serial.print(" ");
    Serial.print(j);
    Serial.print(" ");
    Serial.println(state);
    */

    slot = state & 0xFF;
    symbol = file->inv_f[slot];
    buf[i] = ((int16_t)symbol)-8; // export decompressed byte and re-center
    state = (state >> 8) * file->dist[symbol] + slot - file->cumulative[symbol];
    
    // remap state and ingest byte
    while ((j < file->size) && (state < 256)) {
      /*
      Serial.print("state: ");
      Serial.println(state);
      */
      state = (state << 8) + file->data[j++];
    }
    
    i += 1;
  }
  
  // more to decompress but buffer is full, save checkpoint
  if ((i >= buf_size) && ((j < file->size) || (state != 256))) {
    file->ckpt_state = state;
    file->ckpt_offset = j;
  }

  return i;
}

size_t inflateInt4(tzip* file, int16_t* buf, size_t max_size) {
  size_t till = min(max_size>>1, file->size-file->ckpt_offset);
  /*
  Serial.print("inflateInt4: size ");
  Serial.println(till);
  Serial.print("offset ");
  Serial.println(file->ckpt_offset);
  if (till == 0) {
    Serial.print("Hit max");
    Serial.println(file->ckpt_offset);
  }
  */

  /* TODO: read in 32 bits at a time and align memory to speed up reading
  INFLATING WEIGHTS TAKES 10X MORE TIME CURRENTLY COMPARED TO COMPUTE
  uint32_t buf;
  uint8_t counter = 0;
  if (counter == 0) {
    buf = ()(file->data + i + (file->ckpt_offset));
  }
  */

  for (size_t i = 0; i < till; i++) {
    // restore signed data
    uint8_t read = file->data[i + (file->ckpt_offset)];
    buf[i*2] = (int16_t)((int16_t)read & 15) - 8;
    buf[i*2+1] = (int16_t)((int16_t)read >> 4) - 8;
    
    /*
    if (i*2 % 24 == 0) {
      Serial.println();
    }
    Serial.print(buf[i*2]);
    Serial.print(" ");
    Serial.print(buf[i*2+1]);
    Serial.print(" ");
    delay(50);
    */
  }
  file->ckpt_offset += till;
  //Serial.println(file->ckpt_offset);
  return till<<1;
}

size_t inflateInt8(tzip file, int16_t* buf, size_t max_size) {
  size_t till = min(max_size, file.size-file.ckpt_offset);

  for (size_t i = 0; i < till; i++) {
    buf[i] = file.data[i+file.ckpt_offset];
  }
  file.ckpt_offset += till;
  return till;
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  x = (int16_t*)malloc(X_SIZE * sizeof(int16_t));
  w = (int16_t*)malloc(X_SIZE * sizeof(int16_t));
  h1 = (int16_t*)malloc(H1_SIZE * sizeof(int16_t));
  h2 = (int16_t*)malloc(H2_SIZE * sizeof(int16_t));
  h3 = (int16_t*)malloc(H3_SIZE * sizeof(int16_t));
  if (x == NULL || w == NULL || h1 == NULL || h2 == NULL || h3 == NULL) {
    while (true) {
      Serial.println("malloc failed!");
      delay(1000);
    }
  }
}

void fc(tzip *file, int16_t M, int16_t* w, const int8_t* b, int16_t* x, int16_t* y, size_t x_size, size_t y_size) {
  for (int i = 0; i < y_size; i++) {
    inflateInt4(file, w, x_size);
    int16_t result;
    dsps_dotprod_s16(x, w, &result, x_size, 15);
    y[i] = (((int32_t)result * (int32_t)M)>>16) + ((int32_t)b[i]);
  }
}

void fcZip(tzip *file, int16_t M, int16_t* w, const int8_t* b, int16_t* x, int16_t* y, size_t x_size, size_t y_size) {
  for (int i = 0; i < y_size; i++) {
    unzip(file, w, x_size);
    int16_t result;
    dsps_dotprod_s16(x, w, &result, x_size, 15);
    y[i] = (((int32_t)result * (int32_t)M)>>16) + ((int32_t)b[i]);
  }
}

void fcRELU(tzip *file, int16_t M, int16_t* w, const int8_t* b, int16_t* x, int16_t* y, size_t x_size, size_t y_size, uint16_t S) {
  uint32_t startCounter, endCounter, inflateCounter, computeCounter;
  inflateCounter = 0;
  computeCounter = 0;

  int16_t result;

  for (int i = 0; i < y_size; i++) {
    asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
    inflateInt4(file, w, x_size);
    asm volatile("esync; rsr %0,ccount":"=a" (endCounter));
    inflateCounter += endCounter - startCounter;

    
    asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
    if (dsps_dotprod_s16(x, w, &result, x_size, 15-4) != 0) {
      Serial.print("ESP_ERR_low occured at ");
      Serial.println(i);
    }
    
    /*
    Serial.print("y[");
    Serial.print(i);
    Serial.print("] = ");
    Serial.print(x[0]);
    Serial.print(" * ");
    Serial.print(w[0]);
    Serial.print(" + ");
    Serial.print(x[1]);
    Serial.print(" * ");
    Serial.print(w[1]);
    Serial.print(" + ");
    Serial.print(x[2]);
    Serial.print(" * ");
    Serial.print(w[2]);
    Serial.print(" + ... + ");
    Serial.print(x[x_size-2]);
    Serial.print(" * ");
    Serial.print(w[x_size-2]);
    Serial.print(" + ");
    Serial.print(x[x_size-1]);
    Serial.print(" * ");
    Serial.print(w[x_size-1]);
    Serial.println();
    delay(50);
    */

    // Old = [16. 0] * [ 0.16] = [16.16]
    // New = [16. 0]*2^1 * [ 0.16] = [16.16]*2^1
    y[i] = ( ( (((int32_t)result)<<4) * (int32_t)M) >> 16) + ((int32_t)b[i]);
    
    /*
    Serial.print(y[i]);
    Serial.print(" = ((");
    Serial.print(result);
    Serial.print(" * ");
    Serial.print(M);
    Serial.print(")>>16) + ");
    Serial.print(b[i]);
    Serial.print(";");
    Serial.println();
    delay(100);
    */
    
    if (y[i] > 6*S) {
      y[i] = 6*S;
    } else if (y[i] < 0) {
      y[i] = 0;
    }
    asm volatile("esync; rsr %0,ccount":"=a" (endCounter));
    computeCounter += endCounter - startCounter;
  }
  Serial.print("Inflate time: ");
  Serial.println(inflateCounter);
  Serial.print("Compute time: ");
  Serial.println(computeCounter);
  Serial.print("Layer time: ");
  Serial.println(inflateCounter + computeCounter);
}

void fcRELUZip(tzip *file, int16_t M, int16_t* w, const int8_t* b, int16_t* x, int16_t* y, size_t x_size, size_t y_size, uint16_t S) {
  uint32_t startCounter, endCounter, inflateCounter, computeCounter;
  inflateCounter = 0;
  computeCounter = 0;

  int16_t result;

  for (int i = 0; i < y_size; i++) {
    asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
    unzip(file, w, x_size);
    asm volatile("esync; rsr %0,ccount":"=a" (endCounter));
    inflateCounter += endCounter - startCounter;
    
    asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
    if (dsps_dotprod_s16(x, w, &result, x_size, 15-4) != 0) {
      Serial.print("ESP_ERR_low occured at ");
      Serial.println(i);
    }

    // Old = [16. 0] * [ 0.16] = [16.16]
    // New = [16. 0]*2^1 * [ 0.16] = [16.16]*2^1
    y[i] = ( ( (((int32_t)result)<<4) * (int32_t)M) >> 16) + ((int32_t)b[i]);
    
    if (y[i] > 6*S) {
      y[i] = 6*S;
    } else if (y[i] < 0) {
      y[i] = 0;
    }
    asm volatile("esync; rsr %0,ccount":"=a" (endCounter));
    computeCounter += endCounter - startCounter;
  }
  Serial.print("Inflate time: ");
  Serial.println(inflateCounter);
  Serial.print("Compute time: ");
  Serial.println(computeCounter);
  Serial.print("Layer time: ");
  Serial.println(inflateCounter + computeCounter);
}

void printArray(int16_t *arr, size_t size, size_t cutoff) {
  for (size_t i = 0; i < size; i++) {
    Serial.print(arr[i]);
    Serial.print(" ");
    if ((i+1) % 18 == 0) {
      Serial.println();
      delay(100);
    }
  }
}

void cifar_normal() {
  tzip file;

  for (int i = 0; i < X_SIZE; i++) {
    x[i] = img[i];
  }

  Serial.println("DIMS: ");
  Serial.println(X_SIZE);
  Serial.println(H1_SIZE);
  Serial.println(H2_SIZE);
  Serial.println(H3_SIZE);

  uint32_t startCounter, counter;

  Serial.println("Layer 1: ");
  file.data = fc1_w;
  file.size = sizeof(fc1_w);
  file.ckpt_offset = 0;
  fcRELU(&file, M1, w, fc1_b, x, h1, X_SIZE, H1_SIZE, S2);
  Serial.println("Results: ");
  //printArray(h1, H1_SIZE, 18);

  Serial.println("Layer 2: ");
  file.data = fc2_w;
  file.size = sizeof(fc2_w);
  file.ckpt_offset = 0;
  fcRELU(&file, M2, w, fc2_b, h1, h2, H1_SIZE, H2_SIZE, S3);
  Serial.println("Results: ");
  //printArray(h2, H2_SIZE, 18);

  Serial.println("Layer 3: ");
  file.data = fc3_w;
  file.ckpt_offset = 0;
  fc(&file, M3, w, fc3_b, h2, h3, H2_SIZE, H3_SIZE);
  Serial.println("Results: ");
  printArray(h3, H3_SIZE, 18);
}

void inv_f_int4(tzip* file) {  
  file->cumulative[0] = 0;
  for (int k = 1; k < 16; k++) {
    file->cumulative[k] = file->cumulative[k-1] + file->dist[k-1];
  }
  
  int y = 0;
  for (int x = 0; x < 256; x++) {
    if ((x < file->cumulative[y]) || (y >= 16)) {
      file->inv_f[x] = y-1;
    } else {
      y += 1;
      file->inv_f[x] = y-1;
    }
  }
}

void check_weights() {
  int16_t *zipBuf, *buf;
  tzip zipFile, file;
  Serial.println("Layer 1: ");

  uint8_t *inv_f, *cumulative;
  inv_f = (uint8_t*)malloc(256);
  cumulative = (uint8_t*)malloc(16);

  zipBuf = (int16_t*)malloc(H1_SIZE * sizeof(int16_t));
  buf = (int16_t*)malloc(H1_SIZE * sizeof(int16_t));

  if (cumulative == NULL || inv_f == NULL || zipBuf == NULL || buf == NULL) {
    while (true) {
      Serial.println("MALLOC in check_weights failed!");
      delay(5000);
    }
  }

  zipFile.data = fc1_w_d;
  zipFile.inv_f = inv_f;
  zipFile.cumulative = cumulative;
  zipFile.dist = fc1_w_f;
  zipFile.size = sizeof(fc1_w_d);
  zipFile.ckpt_offset = 0;

  //populate the lookup table
  Serial.println("Populating frequency table");
  inv_f_int4(&zipFile);

  file.data = fc1_w;
  file.inv_f = NULL;
  file.size = sizeof(fc1_w);
  file.ckpt_offset = 0;
  
  // unzip H1_SIZE for X_SIZE times, compare it to uncompressed version
  for (int i = 0; i < X_SIZE; i++) {
    size_t inflateSize = inflateInt4(&file, buf, H1_SIZE);
    size_t unzipSize = unzip(&zipFile, zipBuf, H1_SIZE);

    for (int j = 0; j < H1_SIZE; j++) {
      if (zipBuf[j] != buf[j]) {
        Serial.print(j);
        Serial.print(": ");
        Serial.print(zipBuf[j]);
        Serial.print(" != ");
        Serial.println(buf[j]);
      }
    }
  }
  
  free(inv_f);
  free(cumulative);
  free(zipBuf);
  free(buf);
}

void cifar_compressed() {
  tzip file;
  Serial.println("Layer 1: ");

  uint8_t *inv_f, *cumulative;
  inv_f = (uint8_t*)malloc(256);
  cumulative = (uint8_t*)malloc(16);

  if (cumulative == NULL || inv_f == NULL) {
    while (true) {
      Serial.println("MALLOC in cifar_compressed failed!");
      delay(5000);
    }
  }

  file.data = fc1_w_d;
  file.inv_f = inv_f;
  file.cumulative = cumulative;
  file.dist = fc1_w_f;
  file.size = sizeof(fc1_w_d);
  file.ckpt_offset = 0;
  inv_f_int4(&file);
  
  fcRELUZip(&file, M1, w, fc1_b, x, h1, X_SIZE, H1_SIZE, S2);
  Serial.println("Results: ");
  //printArray(h1, H1_SIZE, 18);

  Serial.println("Layer 2: ");
  file.data = fc2_w_d;
  file.dist = fc2_w_f;
  file.size = sizeof(fc2_w_d);
  file.ckpt_offset = 0;
  inv_f_int4(&file);
  fcRELUZip(&file, M2, w, fc2_b, h1, h2, H1_SIZE, H2_SIZE, S3);
  Serial.println("Results: ");
  //printArray(h2, H2_SIZE, 18);

  Serial.println("Layer 3: ");
  file.data = fc3_w_d;
  file.dist = fc3_w_f;
  file.size = sizeof(fc3_w_d);
  file.ckpt_offset = 0;
  inv_f_int4(&file);
  fcZip(&file, M3, w, fc3_b, h2, h3, H2_SIZE, H3_SIZE);
  Serial.println("Results: ");
  printArray(h3, H3_SIZE, 18);
  
  free(inv_f);
  free(cumulative);
}

// This is the compressed ino, start with the not compressed version so at least we have something
// Compression might take a while...
void loop() {
  // put your main code here, to run repeatedly:
  uint32_t startCounter, counter;
  //asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
  //cifar_normal();
  //asm volatile("esync; rsr %0,ccount":"=a" (counter));
  //Serial.print("Total Time: ");
  //Serial.println(counter - startCounter);

  asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
  check_weights();
  asm volatile("esync; rsr %0,ccount":"=a" (counter));
  Serial.print("Weight check Time: ");
  Serial.println(counter - startCounter);

  asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
  cifar_compressed();
  asm volatile("esync; rsr %0,ccount":"=a" (counter));
  Serial.print("Compressed Time: ");
  Serial.println(counter - startCounter);

  while (true) {
    delay(5000);
    Serial.println("Full stop");
  }
}

// ~2,660,000 cycles
