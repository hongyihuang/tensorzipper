#include <stdlib.h>
#include "./cifar-10-fc-runtime.h"

void setup() {  
  Serial.begin(115200);
  delay(5000);
  Serial.println("Begin");
  cifar_setup();
}

void loop() {
  // put your main code here, to run repeatedly:
  uint32_t startCounter, counter;
  /*
  asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
  cifar_normal();
  asm volatile("esync; rsr %0,ccount":"=a" (counter));
  Serial.print("Total Time: ");
  Serial.println(counter - startCounter);
  */

  asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
  check_weights();
  asm volatile("esync; rsr %0,ccount":"=a" (counter));
  Serial.print("Weight check Time: ");
  Serial.println(counter - startCounter);
  printms(TREAD_TIME_ID);
  
  /*
  asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
  cifar_compressed();
  asm volatile("esync; rsr %0,ccount":"=a" (counter));
  Serial.print("Compressed Time: ");
  Serial.println(counter - startCounter);
  */

  while (true) {
    delay(5000);
    Serial.println("Full stop");
  }
}

// ~2,660,000 cycles
//Normal: -7 -8  1 7 -5 -6 -4 -7 -6 0
//Zipped: -7 -8  1 7 -5 -6 -4 -7 -6 0
//
