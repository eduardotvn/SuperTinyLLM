#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

void setup_tensor_view(Tensor *t, float *data, int d0, int d1, int d2) {
    t->data = data;
    t->ndim = 0;
    
    if (d0 > 0) {
        t->shape[0] = d0; t->ndim++;
    }
    if (d1 > 0) {
        t->shape[1] = d1; t->ndim++;
    }
    if (d2 > 0) {
        t->shape[2] = d2; t->ndim++; 
    }
    
    for (int i = t->ndim; i < 4; i++) {
        t->shape[i] = 1;}

    t->size = 1;

    for (int i = 0; i < t->ndim; i++) {
        t->size *= t->shape[i];
    }
}

void malloc_run_state(RunState *s, Config *p) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads)/(p->n_heads);


    int shape_x[] = {dim};
    s->x = tensor_create(1, shape_x);
    s->xb = tensor_create(1, shape_x);
    s->hb = tensor_create(1, (int[]){hidden_dim});

    s->q = tensor_create(1, shape_x);
    s->k = tensor_create(1, (int[]){kv_dim}); 
    s->v = tensor_create(1, (int[]){kv_dim});
    
    s->logits = tensor_create(1, (int[]){p->vocab_size});

    s->att = tensor_create(1, (int[]){p->n_heads, p->seq_len});

    int kv_cache_size = p->n_layers * p->seq_len * kv_dim;
    s->key_cache = (float *)calloc(kv_cache_size, sizeof(float));
    s->value_cache = (float *)calloc(kv_cache_size, sizeof(float));

    if (!s->key_cache || !s->value_cache) {
        fprintf(stderr, "Fatal error: Failed allocating KV cache.\n");
        exit(1);
    }
}

void free_run_state(RunState *s) {
    tensor_free(&s->x);
    tensor_free(&s->xb);
    tensor_free(&s->hb);
    tensor_free(&s->q);
    tensor_free(&s->k);
    tensor_free(&s->v);
    tensor_free(&s->att);
    tensor_free(&s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
    int head_size = ( p->dim)/(p->n_heads);
    
    float* (*next)(int) = NULL; 

    setup_tensor_view(&w->token_embedding_table, ptr, p->vocab_size, p->dim, 0);
    ptr += (p->vocab_size)*(p->dim);


    w->rms_att_weight = malloc(p->n_layers * sizeof(Tensor));
    w->wq = malloc(p->n_layers * sizeof(Tensor));
    w->wk = malloc(p->n_layers * sizeof(Tensor));
    w->wv = malloc(p->n_layers * sizeof(Tensor));
    w->wo = malloc(p->n_layers * sizeof(Tensor));
    
    w->rms_ffn_weight = malloc(p->n_layers * sizeof(Tensor));
    w->w1 = malloc(p->n_layers * sizeof(Tensor));
    w->w2 = malloc(p->n_layers * sizeof(Tensor));
    w->w3 = malloc(p->n_layers * sizeof(Tensor));

    
    for (int i = 0; i < p->n_layers; i++) {
        setup_tensor_view(&w->rms_att_weight[i], ptr, p->dim, 0, 0);
        ptr += p->dim;
    }

    for (int i = 0; i < p->n_layers; i++) {
        setup_tensor_view(&w->wq[i], ptr, p->dim, p->dim, 0);
        ptr += p->dim * p->dim;
    }

    for (int i = 0; i < p->n_layers; i++) {
        setup_tensor_view(&w->wk[i], ptr, p->dim, p->dim, 0); 
        ptr += p->dim * p->dim;
    }

    for (int i = 0; i < p->n_layers; i++) {
        setup_tensor_view(&w->wv[i], ptr, p->dim, p->dim, 0); 
        ptr += p->dim * p->dim;
    }

    for (int i = 0; i < p->n_layers; i++) {
        setup_tensor_view(&w->wo[i], ptr, p->dim, p->dim, 0);
        ptr += p->dim * p->dim;
    }

    for (int i = 0; i < p->n_layers; i++) {
        setup_tensor_view(&w->rms_ffn_weight[i], ptr, p->dim, 0, 0);
        ptr += p->dim;
    }

    for (int i = 0; i < p->n_layers; i++) {
        setup_tensor_view(&w->w1[i], ptr, p->hidden_dim, p->dim, 0);
        ptr += p->hidden_dim * p->dim;
    }

    for (int i = 0; i < p->n_layers; i++) {
        setup_tensor_view(&w->w2[i], ptr, p->dim, p->hidden_dim, 0);
        ptr += p->dim * p->hidden_dim;
    }

    for (int i = 0; i < p->n_layers; i++) {
        setup_tensor_view(&w->w3[i], ptr, p->hidden_dim, p->dim, 0);
        ptr += p->hidden_dim * p->dim;
    }

    setup_tensor_view(&w->rms_final_weight, ptr, p->dim, 0, 0);
    ptr += p->dim;

    ptr += p->seq_len * head_size / 2 * 2; 

    if (shared_weights) {
        w->w_cls = w->token_embedding_table; 
    } else {
        setup_tensor_view(&w->w_cls, ptr, p->vocab_size, p->dim, 0);
        ptr += p->vocab_size * p->dim;
    }
}

void build_transformer(Transformer *t, char *weights_path) {
    FILE *file = fopen(weights_path, "rb");
    if (!file) {
        fprintf(stderr, "Error: not possible to open %s\n", weights_path);
        exit(1);
    }

    if (fread(&t->config, sizeof(Config), 1, file) != 1) {
        fprintf(stderr, "Error reading header \n"); exit(1);
    }

    printf("Loading model: layers=%d, dim=%d, vocab=%d\n", 
           t->config.n_layers, t->config.dim, t->config.vocab_size);

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET); 
    
    fseek(file, sizeof(Config), SEEK_SET);
    
    long data_bytes = file_size - sizeof(Config);
    t->data_memory = (float *)malloc(data_bytes);
    
    if (!t->data_memory) {
        fprintf(stderr, "Error: No memory for storing weights (%ld bytes)\n", data_bytes);
        exit(1);
    }

    if (fread(t->data_memory, 1, data_bytes, file) != data_bytes) {
        fprintf(stderr, "Error: Incomplete file reading \n");
        exit(1);
    }
    fclose(file);

    memory_map_weights(&t->weights, &t->config, t->data_memory, 1); 

    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
    free_run_state(&t->state);
    free(t->weights.rms_att_weight);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.rms_ffn_weight);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    free(t->data_memory);
}