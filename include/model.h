#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "config.h"

typedef struct {

    Tensor token_embedding_table; 

    Tensor *rms_att_weight; 
    Tensor *wq;             
    Tensor *wk;             
    Tensor *wv;             
    Tensor *wo;             
    
    Tensor *rms_ffn_weight; 
    Tensor *w1;             
    Tensor *w2;             
    Tensor *w3;             
    Tensor rms_final_weight; 

    Tensor w_cls; 
} TransformerWeights;

typedef struct {
    Tensor x;      
    Tensor xb;     
    Tensor hb;     
    Tensor q;      
    Tensor k;      
    Tensor v;      
    Tensor att;    
    Tensor logits; 
    float *key_cache;   
    float *value_cache; 
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    float *data_memory; 
    int fd;            
} Transformer;

void malloc_run_state(RunState *s, Config *p);
void free_run_state(RunState *s);
void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights);

#endif