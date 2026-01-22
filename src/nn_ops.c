#include "nn_ops.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void nn_rmsnorm(Tensor *out, Tensor *in, Tensor *weight, float eps) {
    if (in->ndim != 2) {
        fprintf(stderr, "Error: RMSNorm needs a 2D tensor [Batch, Dimension]\n");
        exit(1);
    }

    int rows = in->shape[0]; 
    int dimensions = in->shape[1];  

    for (int i = 0; i < rows; i++) {
        float ss = 0.0f;
        for (int j = 0; j < dimensions; j++) {
            float val = tensor_get_2d(in, i, j);
            ss += val * val;
        }

        ss /= dimensions;
        float inv_rms = 1.0f / sqrtf(ss + eps);

        for (int j = 0; j < dimensions; j++) {
            float val = tensor_get_2d(in, i, j);
            float w = weight->data[j]; 
            tensor_set_2d(out, i, j, val * inv_rms * w);
        }
    }
}

void nn_silu(Tensor *t) {
    for (int i = 0; i < t->size; i++) {
        float value = t->data[i];
        float sigmoid = 1.0f / (1.0f + expf(-value));
        t->data[i] = value * sigmoid;
    }
}