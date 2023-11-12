#include "nanotorch.h"
#include "ans.h"
#include <stdlib.h>
#include <stdio.h>

#define RESULT_SHIFT 6

int16_t dotp_8_32_16(int8_t *a, int8_t *b, int n, uint8_t s) {
  int32_t rst = 0;
  size_t i = 0;

  #ifdef ESP32S3
  asm volatile("WUR.ACCX_0 %0; WUR.ACCX_1 %1"::"r"(0), "r"(0));

  asm volatile( 
    "EE.VLD.128.IP q0, %0, 16 \n\t"
    "EE.VLD.128.IP q1, %1, 16 \n\t"
    :
    :"r" (a), "r" (b)
  );
  for (; i <= n>>6; i++) {
    asm volatile( 
      "EE.VLD.128.IP q2, %0, 16 \n\t"
      "EE.VMULAS.S8.ACCX.LD.IP q3, %1, 16, q0, q1 \n\t"
      "EE.VLD.128.IP q4, %0, 16 \n\t"
      "EE.VMULAS.S8.ACCX.LD.IP q5, %1, 16, q2, q3 \n\t"
      "EE.VLD.128.IP q6, %0, 16 \n\t"
      "EE.VMULAS.S8.ACCX.LD.IP q7, %1, 16, q4, q5 \n\t"
      "EE.VLD.128.IP q0, %0, 16 \n\t"
      "EE.VMULAS.S8.ACCX.LD.IP q1, %1, 16, q6, q7 \n\t"
      :
      :"r" (a), "r" (b)
    );
  }
  //asm volatile("RUR.ACCX_0 %0; RUR.ACCX_1 %1":"=r"(rst), "=r"(???):);
  asm volatile("RUR.ACCX_0 %0":"=r"(rst):);
  i = i<<6;
  #else
  // Loop unrolling: Process four elements in a single loop iteration
  // clang will auto-vectorize this
  for (; i <= n - 4; i+=4) {
    rst += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
  }
  #endif
  
  // Handle remaining elements
  for (; i < n; i++) {
    rst += a[i] * b[i];
  }
  rst = rst >> s;
  rst = MIN(rst, 32767);
  return rst;
}

/*
void FC(int8_t *x, Matrix w, int8_t *b, int8_t M, int8_t *rst, int8_t n, uint8_t s) {
    for (int i = 0; i < w.cols; i++) {
        rst[i] = M * (dotp_8_32_16(x, &w.data[i*n], n, s)) + b[i];
    }
}
*/

void fc(tzip *file, int16_t M, int8_t* w, const int8_t* b, int8_t* x, int8_t* y, size_t x_size, size_t y_size, size_t (*zipFunc)(tzip*, int8_t*, size_t)) {
  int32_t result;
  for (int i = 0; i < y_size; i++) {
    zipFunc(file, w, x_size);
    result = dotp_8_32_16(x, w, x_size, RESULT_SHIFT);
    result = (((int32_t)(result<<RESULT_SHIFT) * (int32_t)M)>>16) + ((int32_t)b[i]);
    if (result < -128) {
      result = -128;
    } else if (result > 127) {
      result = 127;
    }
    y[i] = result;
  }
}

void fcRELU(tzip *file, int16_t M, int8_t* w, const int8_t* b, int8_t* x, int8_t* y, size_t x_size, size_t y_size, uint16_t S, size_t (*zipFunc)(tzip*, int8_t*, size_t)) {
  uint32_t startCounter, endCounter, inflateCounter, computeCounter;
  inflateCounter = 0;
  computeCounter = 0;

  int32_t result;
  int8_t maxResult = MIN(127, 6*S);
  
  printf("6*S = %d \n", 6*S);

  for (int i = 0; i < y_size; i++) {
    //asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
    zipFunc(file, w, x_size);
    result = dotp_8_32_16(x, w, x_size, RESULT_SHIFT);
    //asm volatile("esync; rsr %0,ccount":"=a" (endCounter));
    //inflateCounter += endCounter - startCounter;

    //asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
    
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
    result = ( ( (((int32_t)result)<<RESULT_SHIFT) * (int32_t)M) >> 16) + ((int32_t)b[i]);
    
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
    
    if (result > maxResult) {
      result = maxResult;
    } else if (result < 0) {
      result = 0;
    }
    y[i] = result;
    //asm volatile("esync; rsr %0,ccount":"=a" (endCounter));
    //computeCounter += endCounter - startCounter;
  }
  /*
  Serial.print("Inflate time: ");
  Serial.println(inflateCounter);
  Serial.print("Compute time: ");
  Serial.println(computeCounter);
  Serial.print("Layer time: ");
  Serial.println(inflateCounter + computeCounter);
  */
}

void printArr(int8_t *arr, size_t size, size_t cutoff) {
  for (size_t i = 0; i < size; i++) {
    #ifdef ARDUINO
    Serial.print(arr[i]);
    Serial.print(" ");
    #else
    printf("%d ", arr[i]);
    #endif
    if ((i+1) % 18 == 0) {
      #ifdef ARDUINO
      Serial.println();
      delay(100);
      #else
      printf("\n");
      #endif
    }
  }
  PRINT("\n");
}