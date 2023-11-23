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
#if defined(ZIPLUT4)
size_t unzip(tzip* file, int8_t* buf, size_t buf_size) {
    uint16_t state;
    int8_t symbol;
    uint32_t entry;
    // i: counter of decompressed buffer
    // j: counter of compressed bitstream
    size_t i;
    size_t j;

    const uint8_t *data = file->data;
    size_t data_size = file->size;
    uint8_t *inv_f = file->inv_f;
    uint32_t *next = file->next;
    size_t chunksize = buf_size>>UNROLL_BITS;
    uint8_t slot;
    uint8_t symlen;

    for (size_t n = 0; n < UNROLL; n++) {
        j = file->ckpt_offset[n];
        state = file->ckpt_state[n];
        for (i = 0; i < chunksize-5; i += symlen) {
            entry = next[state];
            symlen = entry>>28;
            
            buf[i + n*chunksize] = ((int8_t) ((entry) & 0xF)) - 8;

            for (size_t k = 1; k < symlen; k++)
                buf[i + n*chunksize + k] = ((int8_t) ((entry>>(4*k)) & 0xF)) - 8;
                        
            state = (entry >> 20) & 0xFF;
            state = (state << 8) + data[j++];
        }
        //printf("i = %zu, j = %zu, state = %hu\n", i, j, state);
        
        for (; i < chunksize; i++) {
            slot = state;
            symbol = inv_f[slot];

            state = (state >> 8) * (file->packed_bins[symbol] & 0xFF) + slot - (file->packed_bins[symbol] >> 8);

            buf[i + n*chunksize] = symbol - 8;
            if ((state < 256) && (j < data_size)) {
                state = (state << 8) + data[j++];
            }
        }
        
        file->ckpt_state[n] = state;
        file->ckpt_offset[n] = j;
    }
    
    /*
    for (i = 0; i < chunksize; i++) {
        symbol0 = inv_f[state0 & 0xFF];
        symbol1 = inv_f[state1 & 0xFF];
        symbol2 = inv_f[state2 & 0xFF];
        symbol3 = inv_f[state3 & 0xFF];

        state0 = next_state[state0];
        state1 = next_state[state1];
        state2 = next_state[state2];
        state3 = next_state[state3];

        buf[i] = symbol0 - 8;
        buf[i + chunksize] = symbol1 - 8;
        buf[i + 2*chunksize] = symbol2 - 8;
        buf[i + 3*chunksize] = symbol3 - 8;

        // simplified error checking, as long as it doesn't overflow file->data
        if ((state0 < 256) && (j0 < data_size)) {
            state0 = (state0 << 8) + data[j0++];
        }
        if ((state1 < 256) && (j1 < data_size)) {
            state1 = (state1 << 8) + data[j1++];
        }
        if ((state2 < 256) && (j2 < data_size)) {
            state2 = (state2 << 8) + data[j2++];
        }
        if ((state3 < 256) && (j3 < data_size)) {
            state3 = (state3 << 8) + data[j3++];
        }
    }
    
    // save checkpoint
    file->ckpt_state[0] = state0;
    file->ckpt_offset[0] = j0;
    file->ckpt_state[1] = state1;
    file->ckpt_offset[1] = j1;
    file->ckpt_state[2] = state2;
    file->ckpt_offset[2] = j2;
    file->ckpt_state[3] = state3;
    file->ckpt_offset[3] = j3;

    return i;
    */
    return i;
}

void init_tzip(tzip *file, const uint8_t *data, const uint8_t *dist, size_t size) {
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

    uint8_t symbol;

    uint8_t dist_reg, cum_reg, slot;

    for (uint32_t i = 0; i < 65536u; i++) {
        uint16_t state = i;
        size_t count = 0;
        uint32_t content = 0;

        while (state >= 256) {
            slot = state;
            symbol = file->inv_f[slot];
            content |= symbol<<(count*4);
            
            dist_reg = file->packed_bins[symbol] & 0xFF;
            cum_reg = file->packed_bins[symbol] >> 8;

            state = (state >> 8) * dist_reg + slot - cum_reg;
            count++;
        }
        if (count > 5) {
            printf("Error: >5 symbols per state = %zu\n", count);
            exit(0);
        }

        content |= count<<28;
        content |= ((uint32_t)state)<<20;
        file->next[i] = content;
    }
}
#endif
#if defined(ZIPLUT3)
size_t unzip(tzip* file, int8_t* buf, size_t buf_size) {
    uint16_t state0, state1, state2, state3;
    int8_t symbol0, symbol1, symbol2, symbol3;
    // i: counter of decompressed buffer
    // j: counter of compressed bitstream
    size_t i;
    size_t j0, j1, j2, j3;

    j0 = file->ckpt_offset[0];
    j1 = file->ckpt_offset[1];
    j2 = file->ckpt_offset[2];
    j3 = file->ckpt_offset[3];
    state0 = file->ckpt_state[0];
    state1 = file->ckpt_state[1];
    state2 = file->ckpt_state[2];
    state3 = file->ckpt_state[3];

    const uint8_t *data = file->data;
    size_t data_size = file->size;
    uint8_t *inv_f = file->inv_f;
    uint16_t *next_state = file->next_state;
    size_t chunksize = buf_size>>UNROLL_BITS;
    
    for (i = 0; i < chunksize; i++) {
        symbol0 = inv_f[state0 & 0xFF];
        symbol1 = inv_f[state1 & 0xFF];
        symbol2 = inv_f[state2 & 0xFF];
        symbol3 = inv_f[state3 & 0xFF];

        state0 = next_state[state0];
        state1 = next_state[state1];
        state2 = next_state[state2];
        state3 = next_state[state3];

        buf[i] = symbol0 - 8;
        buf[i + chunksize] = symbol1 - 8;
        buf[i + 2*chunksize] = symbol2 - 8;
        buf[i + 3*chunksize] = symbol3 - 8;

        // simplified error checking, as long as it doesn't overflow file->data
        if ((state0 < 256) && (j0 < data_size)) {
            state0 = (state0 << 8) + data[j0++];
        }
        if ((state1 < 256) && (j1 < data_size)) {
            state1 = (state1 << 8) + data[j1++];
        }
        if ((state2 < 256) && (j2 < data_size)) {
            state2 = (state2 << 8) + data[j2++];
        }
        if ((state3 < 256) && (j3 < data_size)) {
            state3 = (state3 << 8) + data[j3++];
        }
    }
    
    // save checkpoint
    file->ckpt_state[0] = state0;
    file->ckpt_offset[0] = j0;
    file->ckpt_state[1] = state1;
    file->ckpt_offset[1] = j1;
    file->ckpt_state[2] = state2;
    file->ckpt_offset[2] = j2;
    file->ckpt_state[3] = state3;
    file->ckpt_offset[3] = j3;

    return i;
}

void init_tzip(tzip *file, const uint8_t *data, const uint8_t *dist, size_t size) {
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

    uint8_t symbol;

    uint8_t dist_reg, cum_reg, slot;

    for (uint32_t state = 256; state < 65536u; state++) {
        slot = state;
        symbol = file->inv_f[slot];
        dist_reg = file->packed_bins[symbol] & 0xFF;
        cum_reg = file->packed_bins[symbol] >> 8;

        file->next_state[state] = (state >> 8) * dist_reg + slot - cum_reg;
    }
}
#endif

#if defined(ZIPLUT2)
size_t unzip(tzip* file, int8_t* buf, size_t buf_size) {
    uint16_t state = file->ckpt_state;
    int16_t symbol;
    // i: counter of decompressed buffer
    // j: counter of compressed bitstream
    size_t i;
    size_t j = file->ckpt_offset;

    const uint8_t *data = file->data;
    size_t data_size = file->size;
    uint8_t *inv_f = file->inv_f;
    uint16_t *next_state = file->next_state;
    uint32_t t0, t1;
    
    for (i = 0; i < buf_size; i++) {
        #ifdef ESP32S3
        asm volatile( 
            "andi %3, %1, 0xFF \n\t"    //t0 = state & 0xFF
            "add %3, %3, %5 \n\t"       //inv_f + t0
            "l16si %2, %5, 0 \n\t"      //symbol = *t0
            "slli %1, %1, 1 \n\t"       //state = state << 1
            "add %4, %6, %1 \n\t"       //t1 = next_state + state
            "l16ui %1, %4, 0 \n\t"      //state = *t1
            "subi %2, %2, 8 \n\t"       //symbol = symbol - 8;
            "add %3, %8, %10 \n\t"      //t0 = buf + i;
            "s8i %2, %3, 0 \n\t"        //*t0 = symbol;
            ""
            :"=r"(j), "=r"(state), "=r"(symbol), "=r"(t0), "=r"(t1)
            :"r"(inv_f), "r"(next_state), "r"(data_size), "r"(buf), "r"(data), "r"(i)
        );
        #else
        symbol = inv_f[state & 0xFF];
        state = next_state[state];
        buf[i] = symbol - 8;
        #endif

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

    uint8_t symbol;
    // i: counter of decompressed buffer
    // j: counter of compressed bitstream
    size_t j = file->ckpt_offset;

    size_t data_size = file->size;
    uint8_t dist_reg, cum_reg, slot;

    for (uint32_t state = 256; state < 65536u; state++) {
        slot = state;
        symbol = file->inv_f[slot];
        dist_reg = file->packed_bins[symbol] & 0xFF;
        cum_reg = file->packed_bins[symbol] >> 8;

        file->next_state[state] = (state >> 8) * dist_reg + slot - cum_reg;
    }
}
#endif

#if defined(ZIPLUT1)
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
#endif

#if defined(ZIPLUT0)
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

#if defined(ZIPLUT3) || defined(ZIPLUT4)
/*
repeatedly call to unzip the file into buffer
Returns size of inflated buffer size
*/
size_t inflateInt4_to_8(tzip* file, int8_t* buf, size_t max_size) {
    // check last loop boundary conditions satisfies
    size_t till = MIN(max_size>>1, file->size - file->ckpt_offset[0]);

    if (till != (max_size>>1)) {
        return 0;
    } 

    for (size_t i = 0; i < till; i++) {
        register uint64_t read = file->data[i + (file->ckpt_offset[0])];

        buf[i*2] = (int8_t)((int8_t)read & 0xF) - 8;
        buf[i*2+1] = (int8_t)((int8_t)(read >> 4) & 0xF) - 8;
    }

    for (size_t i = 0; i < UNROLL; i++) file->ckpt_offset[i] += till;
    return till<<1;
}

size_t inflateInt8_to_8(tzip* file, int8_t* buf, size_t max_size) {
    size_t till = MIN(max_size, file->size-file->ckpt_offset[0]);

    for (size_t i = 0; i < till; i+=8) {
        *((uint64_t*) (buf + i)) = *((uint64_t*) ((file->data + i + file->ckpt_offset[0])));
    }
    file->ckpt_offset[0] += till;
    return till;
}
#else
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
#endif

#ifdef ZIPLUT1
/* Generates a compressed bitstream for benchmarking. */
void compress(tzip *file, uint8_t *dist, uint8_t *weights, size_t size) {
    uint8_t *data = malloc(size);
    if (data == NULL) {
        printf("generate: malloc data failed!");
        exit(0);
    }

    uint32_t state = 256;
    uint8_t symbol;
    size_t j = 0;

    printf("Encoding...\n");
    for (size_t i = 0; i < size; i++) {
        //printf("%zu-%zu\n", size, i);
        symbol = weights[size-1-i]; // reversed read in
        while (state >= (dist[symbol]<<8)) {
            //printf("%hu %u %u\n", state, dist[symbol], symbol);
            data[j++] = state & 0xFF;
            state = state >> 8;
        }
        //printf("state\n");
        state = ((state/dist[symbol])<<8) + (file->packed_bins[symbol] >> 8) + (state % dist[symbol]); 
    }

    // printf("j = %zu, size = %zu \n", j, size);

    // reverse the buffer by allocating another buffer and reverse it... 
    // plenty of memory usually, just a test so perf doesn't matter
    printf("Reversing...\n");
    uint8_t *data_rev = malloc(j+3);
    if (data_rev == NULL) {
        printf("generate: malloc data_rev failed!");
        exit(0);
    }

    // First 3 bytes is ckpt_state
    data_rev[2] = state & 0xFF;
    data_rev[1] = (state>>8) & 0xFF;
    data_rev[0] = (state>>16) & 0xFF;
    
    file->ckpt_state = state;

    for (size_t i = 0; i < j; i++) {
        data_rev[j-1-i+3] = data[i];
    }

    file->size = j+3;
    file->data = data_rev;

    file->ckpt_offset = 3;
    //file->ckpt_state = (data[0] << 16) + (data[1] << 8) + data[2];
    free(data);
}
#else
/* Generates a compressed bitstream for benchmarking. */
void compress(tzip *file, uint8_t *weights, size_t size) {
    uint8_t *data = malloc(size);
    if (data == NULL) {
        printf("generate: malloc data failed!");
        exit(0);
    }

    uint32_t state = 256;
    uint8_t symbol;
    size_t j = 0;
    //size_t tot = 0;

    // loop over in zigzag order
    printf("Encoding...\n");
    // loop over n thread-blocks
    for (size_t i = 0; i < UNROLL; i++) {
        uint8_t *weights_ptr = weights + size - 1 - i*(file->rows>>UNROLL_BITS);
        //loop over the block
        for (size_t k = 0; k < (size>>UNROLL_BITS)/(file->rows>>UNROLL_BITS); k++) {
            // loop over one row of zigzag pattern
            for (size_t l = 0; l < file->rows>>UNROLL_BITS; l++) {
                symbol = *(weights_ptr--); // reversed read in
                uint8_t dist_reg = file->packed_bins[symbol] & 0xFF;
                while (state >= (dist_reg<<8)) {
                    data[j++] = state & 0xFF;
                    state = state >> 8;
                }
                state = ((state/dist_reg)<<8) + (file->packed_bins[symbol] >> 8) + (state % dist_reg); 
                //tot++;
            }
            // decrement by stride
            weights_ptr -= (file->rows>>UNROLL_BITS) * (UNROLL-1);
        }
        // save the ckpt state and offset
        file->ckpt_offset_init[i] = j;
        file->ckpt_state_init[i] = state;
        //printf("j = %zu, state = %u\n", j, state);
    }

    //printf("tot = %zu\n", tot);

    // reverse the buffer by allocating another buffer and reverse it... 
    printf("Reversing...\n");
    uint8_t *data_rev = malloc(j);
    if (data_rev == NULL) {
        printf("generate: malloc data_rev failed!");
        exit(0);
    }
    // inverting data block
    for (size_t i = 0; i < j; i++) {
        data_rev[j-1-i] = data[i];
    }

    file->size = j;
    file->data = data_rev;
    //for (size_t i = 0; i < UNROLL; i++) printf( "i = %zu, state = %hu, offset = %zu\n", i, file->ckpt_state_init[i], file->ckpt_offset_init[i]);

    // invert checkpoint offset coordinates and swap them
    for (size_t i = 0; i < UNROLL; i++) {
        file->ckpt_offset[UNROLL-1-i] = j - file->ckpt_offset_init[i];
        file->ckpt_state[UNROLL-1-i] = file->ckpt_state_init[i];
    }
    //for (size_t i = 0; i < UNROLL; i++) printf( "i = %zu, state = %hu, offset = %zu\n", i, file->ckpt_state[i], file->ckpt_offset[i]);
    for (size_t i = 0; i < UNROLL; i++) {
        file->ckpt_state_init[i] = file->ckpt_state[i];
        file->ckpt_offset_init[i] = file->ckpt_offset[i];
    }
    //for (size_t i = 0; i < UNROLL; i++) printf( "i = %zu, state = %hu, offset = %zu\n", i, file->ckpt_state_init[i], file->ckpt_offset_init[i]);
    
    free(data);
}
#endif