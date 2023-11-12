#ifndef ANS_H
#define ANS_H
#include <stdlib.h>
#include "arch.h"

typedef struct {
  const uint8_t *data; // [size]
  /* Bins packed to reduce memory access
   * lower 8 bits is PDF (Probability Distribution Function)
   * upper 8 bits is CDF (Cumulative Distribution Function)
   */
  uint16_t packed_bins[16];
  uint8_t inv_f[256];
  size_t size;
  size_t ckpt_offset;
  uint32_t ckpt_state;
} tzip;

typedef struct {
  const uint8_t *data; // [size]
  size_t size;
  size_t ckpt_offset;
  uint32_t ckpt_state;

  uint32_t LUT[256]; // 1KB
  // pack inf_f, dist, cumulative to single uint32_t location
  // invf: 4b * 256 = 1024b or 128B (EE.MOVI.32.A this fits on ESPs3 vector regs)
  // dist: 8b * 16  = 128b
  // cumu: 8b * 16  = 128b
  // Order: dist(8), cumu(8), inv_f(8)
  // have 8 extra bits of 0, but avoids boundary crossing
} tzipLUT;

/*
repeatedly call to unzip the file into buffer
Returns size of deflated buffer size use int8 instead
*/

size_t inflateInt8_to_8(tzip* file, int8_t* buf, size_t max_size);

size_t inflateInt4_to_8(tzip* file, int8_t* buf, size_t max_size);

size_t unzip(tzip* file, int8_t* buf, size_t buf_size);

size_t unzipLUT(tzipLUT* file, int8_t* buf, size_t buf_size);

void init_tzip(tzip *file, const uint8_t *data, const uint8_t *dist, size_t size);

void init_tzipLUT(tzipLUT *file, const uint8_t *data, const uint8_t *dist, size_t size);

/* Experimental 
// DO NOT USE THIS IS HUGE
typedef struct {
  //(2**16)*3 = 196,608 KB
  uint8_t *symbol;
  uint16_t *state;
} stateLUT;

size_t unzipLUT(tzip* file, int16_t* buf, size_t buf_size);

void fillLUT(tzip* file, stateLUT* table);

size_t unzipLUT(tzip* file, int16_t* buf, size_t buf_size) {
  uint32_t state = file->ckpt_state;
  int16_t symbol;
  // i: counter of decompressed buffer
  // j: counter of compressed bitstream
  size_t j = file->ckpt_offset;

  const uint8_t *data = file->data;
  size_t data_size = file->size;
  uint8_t *lut_sym, *lut_state;
  lut_sym = file->LUT->symbol;
  lut_state = file->LUT->state;
  //uint8_t data_buf;

  //uint32_t startCounter, endCounter;

  //while (((j < data_size) || (state != 256)) && (i < buf_size))
  //asm volatile("esync; rsr %0,ccount":"=a" (startCounter));
  int i;
  for (i = 0; i < buf_size; i++) {
    symbol = lut_sym[state];
    state = lut_state[state];

    // profile the array random access time
    // remap state and ingest byte
    while ((j < data_size) && (state < 256)) {
      state = (state << 8) + data[j++];
      //data_buf = data[++j];
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

void fillLUT(tzip* file, LUT* table) {
  uint32_t state = file->ckpt_state;
  uint8_t symbol;
  // i: counter of decompressed buffer
  // j: counter of compressed bitstream
  size_t j = file->ckpt_offset;

  const uint8_t *data;
  uint8_t *inv_f, *cumulative, *dist;
  dist = file->dist;
  inv_f = file->inv_f;
  cumulative = file->cumulative;
  data = file->data;
  size_t data_size = file->size;
  uint8_t dist_reg, cum_reg, slot;

  for (int i = 257; i < 256*256; i++) {
    slot = state;
    symbol = inv_f[slot];
    // test cache latency, store in reg if needed
    dist_reg = dist[symbol];
    cum_reg = cumulative[symbol];

    //buf[i] = symbol-8; // export decompressed byte and re-center

    state = (state >> 8) * dist_reg;
    state += slot - cum_reg;

    table->state[i] = (uint16_t)state;
    table->symbol[i] = symbol;
  }
}

*/

#endif