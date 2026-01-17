#include <stdio.h>
#include <stdlib.h>
#include "tokenizer.h"
#include "tensor.h"

int main() {

    printf("Testing tensors...\n");

    int shape[2] = {3,3};
    Tensor tensor = tensor_create(2, shape); 

    tensor_fill(&tensor, 2);

    tensor_print(&tensor);

    tensor_free(&tensor);

    const char* BIN_PATH = "tokenizer_bin/tokenizer.bin";

    Tokenizer tokenizer; 

    printf("\nTesting tokenizer...\n");

    tokenizer_init(&tokenizer, BIN_PATH, 32000);

    tokenizer_free(&tokenizer);

    return 0;
}
