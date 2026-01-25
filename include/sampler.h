#ifndef SAMPLER_H
#define SAMPLER_H

#include "tensor.h" // <--- ADICIONE ISSO AQUI

typedef struct {
    float temperature;
    float top_p;
    unsigned long long rng_state;
} Sampler;

int sample(Sampler *s, Tensor *logits);

#endif