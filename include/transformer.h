#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "model.h"
#include "tensor.h"
#include "nn_ops.h"

void build_transformer(Transformer *t, char *weights_path);
void free_transformer(Transformer *t);
Tensor* transformer_forward(Transformer *t, int token, int pos);

#endif