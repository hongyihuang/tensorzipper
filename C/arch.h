#ifndef ARCH_H
#define ARCH_H

//#define ARDUINO
//#define ESP32S3

#ifdef ARDUINO
#define PRINT Serial.print
#else
#include <stdio.h>
#include <stdlib.h>
#define PRINT(x) printf("%s", x)
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#endif