#ifndef CONFIG_H
#define CONFIG_H

typedef struct {
    int dim;        
    int hidden_dim; 
    int n_layers;   
    int n_heads;    
    int n_kv_heads; 
    int vocab_size; 
    int seq_len;    
} Config;

#endif
