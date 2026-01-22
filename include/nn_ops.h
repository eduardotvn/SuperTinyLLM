#ifndef NN_OPS_H
#define NN_OPS_H

#include "tensor.h"

void nn_rmsnorm(Tensor *out, Tensor *in, Tensor *weight, float eps);
void nn_silu(Tensor *t);

#endif 