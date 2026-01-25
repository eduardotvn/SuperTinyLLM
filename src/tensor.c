#include <stdlib.h>
#include <stdio.h>
#include "tensor.h"
#include <math.h>

Tensor tensor_create(int ndim, int *shape) {
    Tensor t;
    t.ndim = ndim;
    t.size = 1;

    for (int i = 0; i < ndim; i++) {
        t.shape[i] = shape[i];
        t.size *= shape[i];
    }

    
    for (int i = ndim; i < 4; i++) {
        t.shape[i] = 1;
    }

    t.data = (float *)malloc(sizeof(float) * t.size);

    if (t.data == NULL) {
        fprintf(stderr, "Error: failed allocating memmory for tensor \n");
        exit(1);
    }

    return t;
}

void tensor_free(Tensor *t) {
    if (t->data != NULL) {
        free(t->data);
        t->data = NULL;
    }
    t->size = 0;
    t->ndim = 0;
}

void tensor_fill(Tensor *t, float value) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] = value;
    }
}

void tensor_print(Tensor *t) {
    printf("Tensor(ndim=%d, shape=[", t->ndim);
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("])\n");

    for (int i = 0; i < t->size; i++) {
        printf("%.4f ", t->data[i]);
    }
    printf("\n");
    printf("Size: %d", t->size);
}

void tensor_softmax_rows(Tensor *t) {
    int rows = 1;
    int cols = t->shape[0];

    if (t->ndim == 2) {
        rows = t->shape[0];
        cols = t->shape[1];
    } else if (t->ndim != 1) {
        fprintf(stderr, "Error: softmax expects 1D or 2D tensor\n");
        exit(1);
    }

    float *data = t->data;

    for (int i = 0; i < rows; i++) {
        float *row_ptr = data + (i * cols);

        float max_val = row_ptr[0];
        for (int j = 1; j < cols; j++) {
            if (row_ptr[j] > max_val) {
                max_val = row_ptr[j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = expf(row_ptr[j] - max_val);
            row_ptr[j] = val;
            sum += val;
        }

        float inv_sum = 1.0f / sum;
        for (int j = 0; j < cols; j++) {
            row_ptr[j] *= inv_sum;
        }
    }
}

void tensor_add_inplace(Tensor *a, Tensor *b) {
    for (int i = 0; i < a->size; i++) {
        a->data[i] += b->data[i];
    }
}