#include "ans.h"

/*  Perf analysis
    Zipped time: 102M/3M = 34 cycles per original element
    Normal time: 34M, or 20M in the limit.
    Gives 20/3 = 6.67 cycle, very hard to beat. Possible if on intel cores... but we're on mcu...
    Can we multithead this or do a lookup? (peripheral bound)

    Current state is 
    state_t -> state_t+1, symbols

    // problem   : linear dependency of compute based on original array size
    // solution? : cache next state and symbols (plural) to fast forward, only depends on compressed array time
+-----+-----+-----+-----+
| OPT |  0  |  1  |  2  |
| ESP | 3x  | 3x  | ?x  |
| M 1 | 43  | 28  | 22  |
| ARD |?????|?????|?????|
+-----+-----+-----+-----+
*/

#ifdef ZIPLUT2
size_t unzip(tzip* file, int8_t* buf, size_t buf_size) {
    uint16_t state = file->ckpt_state;
    int16_t symbol;
    // i: counter of decompressed buffer
    // j: counter of compressed bitstream
    size_t i;
    size_t j = file->ckpt_offset;

    const uint8_t *data = file->data;
    size_t data_size = file->size;

    for (i = 0; i < buf_size; i++) {
        symbol = file->inv_f[state & 0xFF];
        state = file->next_state[state];
        buf[i] = symbol - 8;

        if ((state < 256) && (j < data_size)) {
            state = (state << 8) + data[j++];
        }
    }

    // more to decompress but buffer is full, save checkpoint
    file->ckpt_state = state;
    file->ckpt_offset = j;

    return i;
}

void init_tzip(tzip *file, const uint8_t *data, const uint8_t *dist, size_t size) {
    file->data = data;
    file->size = size;
    file->ckpt_state = (data[0] << 16) + (data[1] << 8) + data[2];
    file->ckpt_offset = 3;

    // {0[8:0], dist[0][8:0]}
    uint16_t packed_bins[16];
    packed_bins[0] = dist[0];
    uint8_t last_dist = dist[0];
    uint8_t cumulative = 0;
    for (size_t k = 1; k < 16; k++) {
        cumulative += last_dist;
        last_dist = dist[k];
        packed_bins[k] = last_dist | (cumulative<<8);
    }

    uint32_t y = 0;
    for (size_t x = 0; x < 256; x++) {
        cumulative = packed_bins[y] >> 8;
        if (!((x < cumulative) || (y >= 16))) {
            y += 1;
        }
        file->inv_f[x] = y-1;
    }

    uint8_t symbol;
    // i: counter of decompressed buffer
    // j: counter of compressed bitstream
    size_t j = file->ckpt_offset;

    size_t data_size = file->size;
    uint8_t dist_reg, cum_reg, slot;

    for (uint32_t state = 256; state < 65536u; state++) {
        slot = state;
        symbol = file->inv_f[slot];
        dist_reg = packed_bins[symbol] & 0xFF;
        cum_reg = packed_bins[symbol] >> 8;

        file->next_state[state] = (state >> 8) * dist_reg + slot - cum_reg;
    }
}

#elif ZIPLUT
// ~380 cycles on ESP32s3, 28ms on M1, slightly faster
size_t unzip(tzip* file, int8_t* buf, size_t buf_size) {    
    uint16_t state = file->ckpt_state;
    int8_t symbol;

    size_t i;
    size_t j = file->ckpt_offset;

    const uint8_t *data = file->data;
    size_t data_size = file->size;
    uint8_t slot;
    uint32_t pack_reg;
    
    // Active registers: 10
    // Total available registers: 16 (-6 for keeping purposes)
    // i, j, data, buf_size, data_size, file, slot, state, buf, scratch
    for (i = 0; i < buf_size; i++) {
        slot = state;
        pack_reg = file->LUT[slot];
        symbol = (pack_reg >> 16) & 0xF;
        buf[i] = symbol-8; // re-center (compiler uses offset) & write to buffer

        // (state >> 8) * dist + slot - cumulative
        state = (state >> 8) * (pack_reg & 0xFF) + slot - ((pack_reg >> 8) & 0xFF);

        // remap state and ingest byte
        while ((state < 256) && (j < data_size)) {
            state = (state << 8) + data[j++];
        }
    }

    file->ckpt_state = state;
    file->ckpt_offset = j;

    return i;
}

void init_tzip(tzip *file, const uint8_t *data, const uint8_t *dist, size_t size) {
    file->data = data;
    file->size = size;
    file->ckpt_state = (data[0] << 16) + (data[1] << 8) + data[2];
    file->ckpt_offset = 3;
    
    uint16_t packed_bins[16];
    
    // {0[8:0], dist[0][8:0]}
    packed_bins[0] = dist[0];
    uint8_t last_dist = dist[0];
    uint8_t cumulative = 0;
    for (size_t k = 1; k < 16; k++) {
        cumulative += last_dist;
        last_dist = dist[k];
        packed_bins[k] = last_dist | (cumulative<<8);
    }
    
    uint32_t y = 0;
    for (size_t x = 0; x < 256; x++) {
        cumulative = packed_bins[y] >> 8;
        if (!((x < cumulative) || (y >= 16))) {
            y += 1;
        }
        file->LUT[x] = packed_bins[y-1] | ((y-1)<<16);
    }
}
#else
/*
repeatedly call to unzip the file into buffer
Returns size of deflated buffer size
use int8 instead
*/
// 397-440 cycles on ESP32 (depending on cache...)
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
        // perform decoding
        slot = state;
        symbol = file->inv_f[slot];
        pack_reg = file->packed_bins[symbol];

        buf[i] = symbol-8; // re-center (compiler uses offset) & write to buffer

        // (state >> 8) * dist + slot - cumulative
        state = (state >> 8) * (pack_reg & 0xFF) + slot - ((pack_reg >> 8) & 0xFF);

        // remap state and ingest byte
        while ((state < 256) && (j < data_size)) {
            state = (state << 8) + data[j++];
        }
    }
    //asm volatile("esync; rsr %0,ccount":"=a" (endCounter));
    //counter += endCounter - startCounter;
    
    // more to decompress but buffer is full, save checkpoint
    /**/
    //if ((i >= buf_size) && ((j < data_size) || (state != 256))) {
    file->ckpt_state = state;
    file->ckpt_offset = j;
    //}

    return i;
}

// 488 cycles on ESP32, not latency problem since cache and prefetcher is on, it made things slower
size_t unzipBuf(tzip* file, int8_t* buf, size_t buf_size) {
    uint16_t state = file->ckpt_state;
    int8_t symbol;

    size_t j = file->ckpt_offset;
    size_t buf_i = 0;
    size_t buf_j = 0;

    const size_t BUF_SIZE = 128;
    static uint8_t data_buf[BUF_SIZE];

    const uint8_t *data = file->data;
    size_t data_size = file->size;
    uint8_t slot;
    uint16_t pack_reg;

    size_t i;
    for (i = 0; i < buf_size; i++) {
        slot = state;
        symbol = file->inv_f[slot];
        pack_reg = file->packed_bins[symbol];

        buf[i] = symbol-8;

        state = (state >> 8) * (pack_reg & 0xFF) + slot - ((pack_reg >> 8) & 0xFF);

        if (buf_j >= buf_i) {
            buf_i = 0;
            buf_j = 0;
            while ((buf_i < BUF_SIZE) && (j < data_size) ) {
                data_buf[buf_i++] = data[j++];
            }
        }

        if ((state < 256) && (buf_j < buf_i))
            state = (state << 8) + data_buf[buf_j++];
    }
    j = j - BUF_SIZE + buf_j;

    file->ckpt_state = state;
    file->ckpt_offset = j;

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
    
    uint32_t y = 0;
    for (size_t x = 0; x < 256; x++) {
        cumulative = file->packed_bins[y] >> 8;
        if (!((x < cumulative) || (y >= 16))) {
            y += 1;
        }
        file->inv_f[x] = y-1;
    }
}
#endif

/*
repeatedly call to unzip the file into buffer
Returns size of deflated buffer size
*/
size_t inflateInt4_to_8(tzip* file, int8_t* buf, size_t max_size) {
    size_t till = MIN(max_size>>1, file->size-file->ckpt_offset);

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

        //TODO: this should not do sign extend shift, that's a bug
        buf[i*2] = (int8_t)((int8_t)read & 0xF) - 8;
        buf[i*2+1] = (int8_t)((int8_t)(read >> 4) & 0xF) - 8;
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

/* Generates a compressed bitstream for benchmarking. */
void generate(tzip *file, const uint8_t *dist, size_t size) {
    size_t data_size = 1024*1024; // start with 1MB
    uint8_t *data = malloc(data_size);
    init_tzip(file, data, dist, data_size);

    uint32_t state = 256;
    uint8_t symbol;
    size_t j = 0;

    uint16_t lfsr = 0xACE1u;

    // this should be reversed, but if we're just generating an rng test set it doesn't matter
    for (int i = 0; i < size; i++) {
        // generate symbol according to frequency, has periodic but good enough
        lfsr ^= lfsr >> 7;
        lfsr ^= lfsr << 9;
        lfsr ^= lfsr >> 13;

        lfsr += lfsr & 0xFF;
        symbol = lfsr & 0xF;

        while (state >= (dist[symbol]<<8)) {
            data[j] = state & 0xFF;
            state = state >> 8;
        }
        state = ((state/dist[symbol])<<8) + file->inv_f[symbol] + (state % dist[symbol]); 
    }

    // reverse the buffer
}
