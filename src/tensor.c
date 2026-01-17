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
        fprintf(stderr, "Erro: falha ao alocar memória para tensor\n");
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

float tensor_get_2d(Tensor *t, int i, int j) {
    if (t->ndim != 2) {
        printf("Erro: tensor_get_2d chamado em tensor ndim=%d\n", t->ndim);
        exit(1);
    }

    int rows = t->shape[0];
    int cols = t->shape[1];

    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        printf("Erro: índice fora dos limites (%d, %d)\n", i, j);
        exit(1);
    }

    int index = i * cols + j;
    return t->data[index];
}

void tensor_set_2d(Tensor *t, int i, int j, float value) {
    if (t->ndim != 2) {
        printf("Erro: tensor_set_2d chamado em tensor ndim=%d\n", t->ndim);
        exit(1);
    }

    int rows = t->shape[0];
    int cols = t->shape[1];

    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        printf("Erro: índice fora dos limites (%d, %d)\n", i, j);
        exit(1);
    }

    int index = i * cols + j;
    t->data[index] = value;
}

Tensor tensor_matmul_2d(Tensor *A, Tensor *B) {
    if (A->ndim != 2 || B->ndim != 2) {
        printf("Erro: matmul requer tensores 2D\n");
        exit(1);
    }

    int m = A->shape[0];
    int kA = A->shape[1];
    int kB = B->shape[0];
    int n = B->shape[1];

    if (kA != kB) {
        printf("Erro: dimensões incompatíveis (%d x %d) * (%d x %d)\n",
               A->shape[0], A->shape[1], B->shape[0], B->shape[1]);
        exit(1);
    }

    int shape[2] = {m, n};
    Tensor C = tensor_create(2, shape);
    tensor_fill(&C, 0.0f);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < kA; p++) {
                float a = tensor_get_2d(A, i, p);
                float b = tensor_get_2d(B, p, j);
                sum += a * b;
            }
            tensor_set_2d(&C, i, j, sum);
        }
    }

    return C;
}

void tensor_softmax_rows(Tensor *t) {
    if (t->ndim != 2) {
        printf("Erro: softmax requer tensor 2D\n");
        exit(1);
    }

    int rows = t->shape[0];
    int cols = t->shape[1];

    for (int i = 0; i < rows; i++) {

        float max = tensor_get_2d(t, i, 0);
        for (int j = 1; j < cols; j++) {
            float v = tensor_get_2d(t, i, j);
            if (v > max) max = v;
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float e = expf(tensor_get_2d(t, i, j) - max);
            tensor_set_2d(t, i, j, e);
            sum += e;
        }

        for (int j = 0; j < cols; j++) {
            float v = tensor_get_2d(t, i, j) / sum;
            tensor_set_2d(t, i, j, v);
        }
    }
}

void tensor_add_inplace(Tensor *a, Tensor *b) {
    for (int i = 0; i < a->size; i++) {
        a->data[i] += b->data[i];
    }
}