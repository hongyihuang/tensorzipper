#include "ans.h"

/*
repeatedly call to unzip the file into buffer
Returns size of deflated buffer size
use int8 instead
*/
size_t unzip(tzip* file, int8_t* buf, size_t buf_size) {
    // initialize pointers and states
    // while loop decompress
    // filled buffer
    
    uint32_t state = file->ckpt_state;
    int8_t symbol;
    // i: counter of decompressed buffer
    // j: counter of compressed bitstream
    size_t j = file->ckpt_offset;

    const uint8_t *data = file->data;
    size_t data_size = file->size;
    uint8_t slot;
    uint16_t pack_reg;

    //uint32_t startCounter, endCounter;

    //while (((j < data_size) || (state != 256)) && (i < buf_size))
    //asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
    size_t i;
    for (i = 0; i < buf_size; i++) {
        /*
        Serial.print("i, j, state: ");
        Serial.print(i);
        Serial.print(" ");
        Serial.print(j);
        Serial.print(" ");
        Serial.println(state);
        */

        // perform decoding
        /* Perf analysis
        Zipped time: 102M/3M = 34 cycles per original element
        Normal time: 34M, or 20M in the limit.
        Gives 20/3 = 6.67 cycle, very hard to beat. Possible if on intel cores... but we're on mcu...
        Can we multithead this or do a lookup? (peripheral bound)

        Current state is 
        state_t -> state_t+1, symbols
        
        // problem   : linear dependency of compute based on original array size
        // solution? : cache next state and symbols (plural) to fast forward, only depends on compressed array time
        */
        slot = state;
        symbol = file->inv_f[slot];
        pack_reg = file->packed_bins[symbol];

        buf[i] = symbol-8; // re-center (compiler uses offset) & write to buffer

        // (state >> 8) * dist + slot - cumulative
        state = (state >> 8) * (pack_reg & 0xFF) + slot - ((pack_reg >> 8) & 0xFF);

        // profile the array random access time
        // remap state and ingest byte
        while ((j < data_size) && (state < 256)) {
            /*
            Serial.print("state: ");
            Serial.println(state);
            */
            // + data[j++]
            state = (state << 8) + data[j++];
        }
    }
    //asm volatile("esync; rsr %0,ccount":"=a" (endCounter));
    //counter += endCounter - startCounter;
    
    // more to decompress but buffer is full, save checkpoint
    if ((i >= buf_size) && ((j < data_size) || (state != 256))) {
        file->ckpt_state = state;
        file->ckpt_offset = j;
    }

    return i;
}

// Abaondon now, benchmark decompresion on scratchpad
size_t unzipLUT(tzipLUT* file, int8_t* buf, size_t buf_size) {
    // initialize pointers and states
    // while loop decompress
    // filled buffer
    
    uint32_t state = file->ckpt_state;
    int8_t symbol;
    // i: counter of decompressed buffer
    // j: counter of compressed bitstream
    size_t j = file->ckpt_offset;

    const uint8_t *data = file->data;
    size_t data_size = file->size;
    uint8_t slot;
    uint32_t pack_reg;

    //uint32_t startCounter, endCounter;

    //while (((j < data_size) || (state != 256)) && (i < buf_size))
    //asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
    size_t i;
    // Active registers: 9
    // Total available registers: 16 (-6 for keeping purposes)
    // i, j, data, buf_size, data_size, file, slot, state, scratch
    for (i = 0; i < buf_size; i++) {
        slot = state;
        pack_reg = file->LUT[slot];
        symbol = pack_reg & 0xF;
        buf[i] = symbol-8; // re-center (compiler uses offset) & write to buffer

        // (state >> 8) * dist + slot - cumulative
        state = (state >> 8) * ((pack_reg >> 8) & 0xFF);
        state += slot - ((pack_reg >> 16) & 0xFF);

        // remap state and ingest byte
        while ((j < data_size) && (state < 256)) {
            state = (state << 8) + data[j++];
        }
    }
    //asm volatile("esync; rsr %0,ccount":"=a" (endCounter));
    //counter += endCounter - startCounter;
    
    // more to decompress but buffer is full, save checkpoint
    if ((i >= buf_size) && ((j < data_size) || (state != 256))) {
        file->ckpt_state = state;
        file->ckpt_offset = j;
    }

    return i;
}

void init_tzip(tzip *file, const uint8_t *data, const uint8_t *dist, size_t size) {
    file->data = data;
    file->size = size;
    file->ckpt_state = (data[0] << 16) + (data[1] << 8) + data[2];
    file->ckpt_offset = 3;
    
    // {0[8:0], dist[0][8:0]}
    file->packed_bins[0] = dist[0];
    uint8_t last_dist = dist[0];
    uint8_t cumulative = 0;
    for (size_t k = 1; k < 16; k++) {
        cumulative += last_dist;
        last_dist = dist[k];
        file->packed_bins[k] = last_dist | (cumulative<<8);
    }
    
    int y = 0;
    for (int x = 0; x < 256; x++) {
        cumulative = file->packed_bins[y] >> 8;
        if ((x < cumulative) || (y >= 16)) {
            file->inv_f[x] = y-1;
        } else {
            y += 1;
            file->inv_f[x] = y-1;
        }
    }
}

void init_tzipLUT(tzipLUT *file, const uint8_t *data, const uint8_t *dist, size_t size) {
    file->data = data;
    //uint8_t cumulative[16];

    file->size = size;
    file->ckpt_state = (data[0] << 16) + (data[1] << 8) + data[2];
    file->ckpt_offset = 3;

    /*
    cumulative[0] = 0;
    for (int k = 1; k < 16; k++) {
        cumulative[k] = cumulative[k-1] + dist[k-1];
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

    for (int i = 0; i < 256; i++) {
        file->LUT;
    }
    */
}

/*
repeatedly call to unzip the file into buffer
Returns size of deflated buffer size
*/
size_t inflateInt4_to_8(tzip* file, int8_t* buf, size_t max_size) {
  size_t till = MIN(max_size>>1, file->size-file->ckpt_offset);
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

    //TODO: this should not do signn extend shift, that's a bug
    buf[i*2] = (int8_t)((int8_t)read & 0xF) - 8;
    buf[i*2+1] = (int8_t)((int8_t)(read >> 4)) - 8;
    
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

size_t inflateInt8_to_8(tzip* file, int8_t* buf, size_t max_size) {
  size_t till = MIN(max_size, file->size-file->ckpt_offset);

  for (size_t i = 0; i < till; i++) {
    buf[i] = file->data[i+file->ckpt_offset];
  }
  file->ckpt_offset += till;
  return till;
}

