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
void tensor_print(Tensor *t);
void tensor_softmax_rows(Tensor *t);
void tensor_add_inplace(Tensor *a, Tensor *b);

#endif