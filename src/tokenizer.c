#include "tokenizer.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

int str_lookup(char *str, Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        if (strcmp(str, t->vocab[i]) == 0) {
            return i;
        }
    }
    return -1;
}

void tokenizer_init(Tokenizer *t, const char *filename, int vocab_size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: couldn't open  %s\n", filename);
        exit(1);
    }

    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));

    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { 
        fprintf(stderr, "Error reading header\n"); exit(1); 
    }

    for (int i = 0; i < vocab_size; i++) {
        if (fread(&t->vocab_scores[i], sizeof(float), 1, file) != 1) { break; }
        
        int len;
        if (fread(&len, sizeof(int), 1, file) != 1) { break; }
        
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { break; }
        t->vocab[i][len] = '\0'; 
    }
    fclose(file);
    printf("Tokenizer succesfully loaded. Vocab Size: %d\n", vocab_size);
}

void tokenizer_free(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
}

char* tokenizer_decode(Tokenizer *t, int token, int prev_token) {
    if (token < 0 || token >= t->vocab_size) {
        return "";
    }

    return t->vocab[token];
}

int tokenizer_encode(Tokenizer *t, const char *text, int *tokens, int n_tokens_max) {

    int n_tokens = 0;
    

    for (const char *c = text; *c != '\0'; c++) {
        if (n_tokens >= n_tokens_max) break;
        
        char buf[2] = {*c, '\0'};
        int id = str_lookup(buf, t);
        
        if (id != -1) {
            tokens[n_tokens++] = id;
        } else {
            fprintf(stderr, "Warning: char '%c' not found in vocab\n", *c);
        }
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < n_tokens - 1; i++) {

            char merge_buf[512]; 
            sprintf(merge_buf, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            
            int id = str_lookup(merge_buf, t);
            if (id != -1) {
                if (t->vocab_scores[id] > best_score) {
                    best_score = t->vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }
        }

        if (best_idx == -1) {
            break; 
        }

        tokens[best_idx] = best_id;
        
        for (int i = best_idx + 1; i < n_tokens - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        n_tokens--;
    }

    return n_tokens;
}