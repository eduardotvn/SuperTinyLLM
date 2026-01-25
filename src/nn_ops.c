#include "nn_ops.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void nn_rmsnorm(Tensor *out, Tensor *in, Tensor *weight, float eps) {
    int rows = 1;
    int dimensions = in->shape[0];

    if (in->ndim == 2) {
        rows = in->shape[0];
        dimensions = in->shape[1];
    } else if (in->ndim > 2) {
        fprintf(stderr, "Error: RMSNorm supports only 1D or 2D tensors\n");
        exit(1);
    }

    float *in_ptr = in->data;
    float *out_ptr = out->data;
    float *w_ptr = weight->data;

    for (int i = 0; i < rows; i++) {
        int offset = i * dimensions;

        float ss = 0.0f;
        for (int j = 0; j < dimensions; j++) {
            float val = in_ptr[offset + j];
            ss += val * val;
        }
        ss /= dimensions;
        
        float inv_rms = 1.0f / sqrtf(ss + eps);

        for (int j = 0; j < dimensions; j++) {
            out_ptr[offset + j] = in_ptr[offset + j] * inv_rms * w_ptr[j];
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

void nn_rope(float* q, float* k, int pos, int head_size, int dim) {
    for (int i = 0; i < dim; i+=2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        
        float q0 = q[i];
        float q1 = q[i+1];
        q[i]   = q0 * fcr - q1 * fci;
        q[i+1] = q0 * fci + q1 * fcr;

        if (k) {
            float k0 = k[i];
            float k1 = k[i+1];
            k[i]   = k0 * fcr - k1 * fci;
            k[i+1] = k0 * fci + k1 * fcr;
        }
    }
}

void matmul(Tensor *out, Tensor *x, Tensor *w) {
    int din = x->shape[0];
    int dout = w->shape[0]; 

    for (int i = 0; i < dout; i++) {
        float val = 0.0f;
        float *w_ptr = w->data + i * din;
        float *x_ptr = x->data;
        for (int j = 0; j < din; j++) {
            val += x_ptr[j] * w_ptr[j];
        }
        out->data[i] = val;
    }
}
