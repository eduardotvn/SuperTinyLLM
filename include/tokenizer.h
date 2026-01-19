#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    char **vocab;          
    float *vocab_scores;  
    int vocab_size;        
    int max_token_length;  
    unsigned char *byte_pieces;
} Tokenizer;

void tokenizer_init(Tokenizer *t, const char *filename, int vocab_size);

int str_lookup(char *str, Tokenizer *t);

void tokenizer_free(Tokenizer *t);

int tokenizer_encode(Tokenizer *t, const char *text, int *tokens, int n_tokens_max);

char* tokenizer_decode(Tokenizer *t, int token, int prev_token);

#endif