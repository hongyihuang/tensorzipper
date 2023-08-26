#include <cstdint>
#include <stdlib.h>
#include <stddef.h>

typedef struct {
    const uint8_t* bitstream;
    const uint8_t* dist;
    const size_t size;

    uint32_t last_state;
    size_t last_pointer;
} TensorZip;

/* Decompress fills buffer from bitstream up to size. 
 * Returns 0 if ended normally, 1 if something is wrong with the buffer.
 */
int decompress(TensorZip data, uint8_t* buf, size_t size, bool restart);

