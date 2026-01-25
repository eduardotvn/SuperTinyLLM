#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char *argv[]) {

    #ifdef _WIN32
    SetConsoleOutputCP(65001);
    #endif

    char *weights_path = "model_bin/stories15M.bin";
    char *tokenizer_path = "tokenizer_bin/tokenizer.bin";
    char *prompt = (argc > 1) ? argv[1] : "The little rat went";
    
    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler = {0.8f, 0.9f, (unsigned long long)time(NULL)};

    build_transformer(&transformer, weights_path);
    tokenizer_init(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    
    printf("Prompt: \"%s\"\n", prompt);
    
    int *prompt_tokens = (int *)malloc(sizeof(int) * strlen(prompt) * 2); 
    int num_prompt_tokens = tokenizer_encode(&tokenizer, prompt, prompt_tokens, strlen(prompt) * 2);
    
    printf("STARTING STORY GENERATION: \n\n");

    int token = prompt_tokens[0];
    int pos = 0;
    int max_steps = 500; 
    long start = clock();

    //The printf was eating the first word so i added this print, hehe
    char *first_word = tokenizer_decode(&tokenizer, prompt_tokens[0], 0);
    printf("%s", first_word);
    fflush(stdout);

    while (pos < max_steps) {

        Tensor *logits = transformer_forward(&transformer, token, pos);

        int next_token;
        if (pos < num_prompt_tokens - 1) {
            next_token = prompt_tokens[pos + 1];
        } else {
            next_token = sample(&sampler, logits);
        }

        if (next_token == 2 || next_token == 0) {
            break; 
        }

        char *piece = tokenizer_decode(&tokenizer, next_token, token);
        
        printf("%s", piece);
        fflush(stdout); 
        token = next_token;
        pos++;

        #ifdef _WIN32
        Sleep(90);
        #else
        usleep(ms * 3000); 
        #endif
    }

    printf("\n\n END OF STORY \n");
    
    free(prompt_tokens);
    free_transformer(&transformer);
    tokenizer_free(&tokenizer);
    
    return 0;
}