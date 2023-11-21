#include "arch.h"

//asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
//asm volatile("esync; rsr %0,ccount":"=a" (endCounter));

//static inline 
void beginms(size_t n) {
  #ifdef ESP32S3
  uint32_t counter;
  asm volatile("esync; rsr %0,ccount":"=a" (counter));
  TIMER_TMP[n] = counter;

  #elif ARDUINO
  TIMER_TMP[n] = millis();

  #else
  struct timeval tv;
  gettimeofday(&tv, 0);
  TIMER_TMP[n] = (tv.tv_sec*1e3 + tv.tv_usec/1e3);
  #endif
}

void endms(size_t n) {
  #ifdef ESP32S3
  uint32_t counter;
  asm volatile("esync; rsr %0,ccount":"=a" (counter));
  TIMER[n] += counter - TIMER_TMP[n];

  #elif ARDUINO
  TIMER[n] += millis() - TIMER_TMP[n];
  
  #else
  struct timeval tv;
  gettimeofday(&tv, 0);
  TIMER[n] += (tv.tv_sec*1e3 + tv.tv_usec/1e3) - TIMER_TMP[n];
  #endif
}

void printms(size_t n) {
  sprintf(buf, "%s Time: %lu ms.\n", TIMER_NAME[n], TIMER[n]);
  PRINT(buf);
  TIMER[n] = 0;
}