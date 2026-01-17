#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>

typedef struct {
    float *data;     
    int ndim;        
    int shape[4];    
    int size;      
} Tensor;

Tensor tensor_create(int ndim, int *shape);
void tensor_free(Tensor *t);
void tensor_fill(Tensor *t, float value);
void tensor_print(Tensor *t);
float tensor_get_2d(Tensor *t, int i, int j);
void tensor_set_2d(Tensor *t, int i, int j, float value);
Tensor tensor_matmul_2d(Tensor *A, Tensor *B);
void tensor_softmax_rows(Tensor *t);
void tensor_add_inplace(Tensor *a, Tensor *b);

#endif