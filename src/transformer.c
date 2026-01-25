#include <math.h>
#include <string.h>
#include "transformer.h"
#include "nn_ops.h"

Tensor* transformer_forward(Transformer *t, int token, int pos) {
    Config *p = &t->config;
    TransformerWeights *w = &t->weights;
    RunState *s = &t->state;

    Tensor *x = &s->x;
    Tensor *xb = &s->xb;
    Tensor *hb = &s->hb;
    Tensor *q = &s->q;
    Tensor *k = &s->k;
    Tensor *v = &s->v;
    Tensor *att = &s->att;
    Tensor *logits = &s->logits;
    
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; 

    float *embed_data = w->token_embedding_table.data + (token * dim);
    memcpy(x->data, embed_data, dim * sizeof(float));

    for(int l = 0; l < p->n_layers; l++) {
        
        nn_rmsnorm(xb, x, &w->rms_att_weight[l], 1e-5f);

        matmul(q, xb, &w->wq[l]);
        matmul(k, xb, &w->wk[l]);
        matmul(v, xb, &w->wv[l]);

        nn_rope(q->data, k->data, pos, head_size, dim);

        int loff = l * p->seq_len * kv_dim; 
        int poff = pos * kv_dim;            
        float *k_cache_row = s->key_cache + loff + poff;
        float *v_cache_row = s->value_cache + loff + poff;
        
        memcpy(k_cache_row, k->data, kv_dim * sizeof(float));
        memcpy(v_cache_row, v->data, kv_dim * sizeof(float));

        for (int h = 0; h < p->n_heads; h++) {
            float *q_head = q->data + (h * head_size);
            float *att_head = att->data + (h * p->seq_len);

            for (int t_step = 0; t_step <= pos; t_step++) {
                int kv_head = h / kv_mul;
                float *k_cache_vec = s->key_cache + loff + (t_step * kv_dim) + (kv_head * head_size);

                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q_head[i] * k_cache_vec[i];
                }
                score /= sqrtf((float)head_size);
                att_head[t_step] = score;
            }

            float max_val = -1e10f;
            for (int i = 0; i <= pos; i++) {
                if (att_head[i] > max_val) max_val = att_head[i];
            }
            float sum = 0.0f;
            for (int i = 0; i <= pos; i++) {
                att_head[i] = expf(att_head[i] - max_val);
                sum += att_head[i];
            }
            for (int i = 0; i <= pos; i++) att_head[i] /= sum;

            float *out_head = xb->data + (h * head_size);
            memset(out_head, 0, head_size * sizeof(float));

            for (int t_step = 0; t_step <= pos; t_step++) {
                int kv_head = h / kv_mul;
                float *v_cache_vec = s->value_cache + loff + (t_step * kv_dim) + (kv_head * head_size);
                float a = att_head[t_step];
                for (int i = 0; i < head_size; i++) {
                    out_head[i] += a * v_cache_vec[i];
                }
            }
        }

        matmul(hb, xb, &w->wo[l]); 

        tensor_add_inplace(x, hb);

        nn_rmsnorm(xb, x, &w->rms_ffn_weight[l], 1e-5f);

        matmul(hb, xb, &w->w1[l]); 
        nn_silu(hb); 

        Tensor temp_w3 = *logits; 
        temp_w3.shape[0] = hidden_dim; 
        temp_w3.size = hidden_dim;
        matmul(&temp_w3, xb, &w->w3[l]);

        for (int i = 0; i < hidden_dim; i++) {
            hb->data[i] *= temp_w3.data[i];
        }

        matmul(xb, hb, &w->w2[l]);

        tensor_add_inplace(x, xb);
    }

    nn_rmsnorm(x, x, &w->rms_final_weight, 1e-5f);

    matmul(logits, x, &w->w_cls);

    return logits;
}