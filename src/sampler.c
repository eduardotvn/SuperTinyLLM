#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "sampler.h"
#include "tensor.h"

float random_f32() { 
    return (float)rand() / (float)RAND_MAX; 
}

int sample(Sampler *s, Tensor *logits) {

    if (s->temperature == 0.0f) {
        int max_i = 0;
        float max_p = logits->data[0];
        for (int i = 1; i < logits->size; i++) {
            if (logits->data[i] > max_p) {
                max_p = logits->data[i];
                max_i = i;
            }
        }
        return max_i;
    }

    for (int i = 0; i < logits->size; i++) {
        logits->data[i] /= s->temperature;
    }

    tensor_softmax_rows(logits);

    float coin = random_f32();
    float cdf = 0.0f;
    
    for (int i = 0; i < logits->size; i++) {
        cdf += logits->data[i];
        if (coin < cdf) {
            return i;
        }
    }

    return logits->size - 1; 
    }