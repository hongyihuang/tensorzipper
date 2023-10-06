typedef struct{
    size_t cols;
    int8_t *data;
} Matrix;

int8_t dotp(int8_t *a, int8_t *b, int n) {
    int8_t rst = 0;
    int i;

    // Loop unrolling: Process four elements in a single loop iteration
    for (i = 0; i <= n - 4; i+=4) {
        rst += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
    }

    // Handle remaining elements
    for (; i < n; i++) {
        rst += a[i] * b[i];
    }

    return rst;
}

void FC(int8_t *x, Matrix w, int8_t *b, int8_t M, int8_t *rst, int8_t n) {
    for (int i = 0; i < w.cols; i++) {
        rst[i] = M * (dotp(x, &w.data[i*n], n)) + b[i];
    }
}
