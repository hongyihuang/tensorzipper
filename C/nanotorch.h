#ifndef NANOTORCH_H
#define NANOTORCH_H

#include <stdlib.h>
#include "ans.h"
//#include <stddef.h>

typedef struct{
    size_t cols;
    int8_t *data;
} Matrix;

/*
Dot product int8_t array a and b with length n
Accumulation has int32_t precision 
return a single int16_t number shifted right by s
*/ 
int16_t dotp_8_32_16(int8_t *a, int8_t *b, int n, uint8_t s);

void FC_int8(int8_t *x, Matrix w, int8_t *b, int8_t M, int16_t *rst, int8_t n, uint8_t s);

void fc(tzip *file, int16_t M, int8_t* w, const int8_t* b, int8_t* x, int8_t* y, size_t x_size, size_t y_size, size_t (*zipFunc)(tzip*, int8_t*, size_t));

void fcRELU(tzip *file, int16_t M, int8_t* w, const int8_t* b, int8_t* x, int8_t* y, size_t x_size, size_t y_size, uint16_t S, size_t (*zipFunc)(tzip*, int8_t*, size_t));

void printArr(int8_t *arr, size_t size, size_t cutoff);

#endif