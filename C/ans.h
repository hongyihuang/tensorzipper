#ifndef ANS_H
#define ANS_H
#include <stdlib.h>
#include "arch.h"

#if defined(ZIPLUT4)
// UNROLL cannot be changed without changing code unrolling
#define UNROLL 4
#define UNROLL_BITS 2
// threads can be changed, it will generate more theads
// on embedded this is limited to how many physical cores there is
#define THREADS 1

typedef struct {
    const uint8_t *data; // [size]
    size_t size;
    size_t rows;

    size_t ckpt_offset[THREADS*UNROLL];
    uint16_t ckpt_state[THREADS*UNROLL];

    size_t ckpt_offset_init[THREADS*UNROLL];
    uint16_t ckpt_state_init[THREADS*UNROLL];

    uint16_t packed_bins[16]; // 32B (for compression)
    // 4 bit length, 8 bit state, 5x 4 bit symbol
    uint32_t next[65536]; // 256KB (for decompression)
    uint8_t inv_f[256]; // 0.25KB
} tzip;
#endif
#if defined(ZIPLUT3)
// UNROLL cannot be changed without changing code unrolling
#define UNROLL 4
#define UNROLL_BITS 2
// threads can be changed, it will generate more theads
// on embedded this is limited to how many physical cores there is
#define THREADS 1

typedef struct {
    const uint8_t *data; // [size]
    size_t size;
    size_t rows;

    size_t ckpt_offset[THREADS*UNROLL];
    uint16_t ckpt_state[THREADS*UNROLL];

    size_t ckpt_offset_init[THREADS*UNROLL];
    uint16_t ckpt_state_init[THREADS*UNROLL];

    uint16_t packed_bins[16]; // 32B
    uint16_t next_state[65536]; // 128KB
    uint8_t inv_f[256]; // 0.25KB
} tzip;
#endif
#if defined(ZIPLUT2)
typedef struct {
    const uint8_t *data; // [size]
    size_t size;
    size_t ckpt_offset;
    uint32_t ckpt_state;

    uint16_t packed_bins[16]; // 32B
    uint16_t next_state[65536]; // 128KB
    uint8_t inv_f[256]; // 0.25KB
} tzip;
#endif
#if defined(ZIPLUT1)
typedef struct {
    const uint8_t *data; // [size]
    size_t size;
    size_t ckpt_offset;
    uint32_t ckpt_state;

    uint32_t LUT[256]; // 1KB
    // pack inf_f, dist, cumulative to single uint32_t location
    // dist: 8b * 16  = 128b
    // cumu: 8b * 16  = 128b
    // invf: 4b * 256 = 1024b or 128B (EE.MOVI.32.A this fits on ESPs3 vector regs)
    // Order from low to high: dist(8), cumu(8), inv_f(8) 
    // have 8 extra bits of 0, but avoids boundary crossing
} tzip;
#endif

#if defined(ZIPLUT0)
typedef struct {
    const uint8_t *data; // [size]
    size_t size;
    size_t ckpt_offset;
    uint32_t ckpt_state;

    /* Bins packed to reduce memory access
      * lower 8 bits is PDF (Probability Distribution Function)
      * upper 8 bits is CDF (Cumulative Distribution Function)
      */
    uint16_t packed_bins[16];
    uint8_t inv_f[256];
} tzip;
#endif

/*
repeatedly call to unzip the file into buffer
Returns size of deflated buffer size use int8 instead
*/

size_t inflateInt8_to_8(tzip* file, int8_t* buf, size_t max_size);

size_t inflateInt4_to_8(tzip* file, int8_t* buf, size_t max_size);

size_t unzip(tzip* file, int8_t* buf, size_t buf_size);

void init_tzip(tzip *file, const uint8_t *data, const uint8_t *dist, size_t size);

void compress(tzip *file, uint8_t *data, size_t size);
#endif
