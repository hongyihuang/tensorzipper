#ifndef ARCH_H
#define ARCH_H

//#define ARDUINO
//#define ESP32S3
//#define ZIPLUT
#define ZIPLUT3

#ifdef ARDUINO
#include <Arduino.h>
#include <stdlib.h>
#define PRINT(x) Serial.print(x)
#else
#include <stdio.h>
#include <stdlib.h>
#define PRINT(x) printf("%s", x)
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

const static char* TIMER_NAME[] = {"Test Write", "Test Read", "Inflate", "Zip", "Read", "Dotprod"};
static unsigned long TIMER[16] = {0};
static unsigned long TIMER_TMP[16] = {0};
static char buf[128];

#define TWRITE_TIME_ID 0
#define TREAD_TIME_ID 1
#define INF_TIME_ID 2
#define ZIP_TIME_ID 3
#define READ_TIME_ID 4
#define PROD_TIME_ID 5

void beginms(size_t n);
void endms(size_t n);
void printms(size_t n);

#endif