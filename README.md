# SuperTinyLLM
Super tiny LLM made in C for learning purposes.

Olá! Este é um projeto de LLM minúsculo (bem menor que 1 bilhão de parâmetros, vamos ver) para uma questão de aprendizado. Aqui, estarei ensinando a construir um LLM para que as pessoas entendam como funciona um LLM, qual a "mágica" por trás do código.

Vamos por partes, farei atualizações frequentes neste repositório até que nosso LLM esteja pronto. 

# Planejando nosso LLM

Primeiramente, vamos conceituar um LLM: LLM é um tipo de modelo de inteligência artificial que é construído a partir de transformers, uma arquitetura proposta em 2017 pela Google, no artigo "Attention is all you need". 

Os modelos baseados em transformers se diferem das inteligências artificiais tradicionais pois trazem consigo o conceito de "atenção", atenção essa que indica o quanto cada informação se relaciona com outra.

Por exemplo: O rato roeu a roupa do rei de Roma. 

Quem roeu a roupa do rei de Roma? O rato. Quando o LLM lê "roeu", seus cálculos encontrarão que "roeu" possui um alto índice de relação com rato, logo, ele terá que criar uma estrutura de informação que seja capaz de guardar essas informações. 

Essa estrutura se dá por meio de matrizes de n dimensões, até milhares de dimensões. Para que possamos representar essas matrizes corretamente e iniciarmos a produção do LLM, hoje, trago um pequeno trecho iniciando o conceito de tensores.

Tensores são representações de matrizes de n dimensões. Nele, temos, além do seu conjunto de dados (os números da matriz), o SHAPE do tensor, que é o "formato" da matriz (se ela é 2x3, 3x3, ou até 3x3x3, caso seja de 3 dimensões). Isso é útil pois precisaremos trabalhar com muitas matrizes e sermos capazes de tratá-las, editá-las, operar sobre elas e o programa precisa saber sempre qual é o formato da matriz.

# Dia 17/01/26:

Começo upando o algoritmo do nosso tensor.c e tensor.h. 

Em tensor.h, iniciamos:

typedef struct {
    float *data;     
    int ndim;        
    int shape[4];    
    int size;      
} Tensor;

Esse struct representará nossa matriz. Data é um vetor de floats, ndim é o número de dimensões da matriz e shape indica o formato da matriz (até 4 dimensões, pois faremos um LLM bem pequeno). Size representa o tamanho (quantidade de números) na matriz para fins de cálculo. 

A função tensor_create cria e retorna um novo tensor, utilizando malloc calculado a partir do size.

A função tensor_free é para liberar a memória. Afinal, isso é C, não queremos uma falha complicada.

A função tensor_print é meramente para observamos o tensor a fim de testes. Há um teste em main.c

A função tensor_fill é para inserir números no vetor. É necessário, pois ao criarmos o vetor, ainda não há números nele, apenas indexação de memória.

A função tensor_set_2d é para trabalharmos com vetor de 2 dimensões (nosso cálculos serão apenas com 2 dimensões) e colocaremos no nosso tensor valores baseados em linha x coluna (i x j), como faríamos em programas maiores ou no papel.

A função tensor_get_2d é apenas para retornar o valor com base em (i x j)

A função tensor_matmul_2d é o programa principal da IA! Um simples cálculo de matrizes compatíveis (isto é, as matrizes tem que ter formato axb * bxc, que resulta numa matriz axc)

A função tensor_softmax_rows é um pouco mais complexa. O softmax é uma forma de normalização de dados que funciona melhor do que a normalização usual. Nele, transformamos números de uma linha de matriz em valores entre 0 e 1, com base na sua magnitude. Isso será extremamente útil para calcularmos as probabilidades (afinal, LLMs trabalham em probabilidades, e usam a probabilidade para "prever" a próxima palavra gerada)

A função tensor_add_inplace é para soma de matrizes.

Quaisquer necessidade, alteraremos as funções futuramente. 

# ENGLISH

Hello! This is a tiny LLM project (well below 1 billion parameters, we’ll see) created purely for learning purposes.
Here, I aim to teach how to build an LLM from scratch, so people can truly understand how an LLM works and what the “magic” behind the code actually is.

We will proceed step by step, with frequent updates to this repository until our LLM is complete.

Planning Our LLM

First, let’s define what an LLM is.
An LLM (Large Language Model) is a type of artificial intelligence model built using transformers, an architecture proposed by Google in 2017 in the paper “Attention Is All You Need”.

Transformer-based models differ from traditional AI approaches because they introduce the concept of attention. Attention measures how strongly each piece of information relates to the others.

For example:

The mouse gnawed the clothes of the King of Rome.

Who gnawed the King of Rome’s clothes? The mouse.
When an LLM reads the word “gnawed”, its internal calculations will find that “gnawed” has a strong relationship with “mouse”. Therefore, the model must create an internal structure capable of storing and processing these relationships.

This structure is built using matrices with many dimensions, sometimes hundreds or even thousands of dimensions. To represent and manipulate these matrices correctly, we first need to introduce the concept of tensors.

Tensors

Tensors are representations of n-dimensional matrices.
In addition to storing the data itself (the numbers in the matrix), a tensor also stores its shape, which defines the format of the matrix (for example, 2×3, 3×3, or even 3×3×3 for a 3D tensor).

This is extremely useful because we will work with many matrices, and the program must always know the matrix shape in order to correctly manipulate, edit, and operate on them.

# UPDATE 17/01/26

I start by uploading the implementation of tensor.c and tensor.h.

In tensor.h, we define:

typedef struct {
    float *data;     
    int ndim;        
    int shape[4];    
    int size;      
} Tensor;

This struct represents our matrix.

data is a vector of floats containing the tensor values

ndim is the number of dimensions

shape defines the tensor format (up to 4 dimensions, since this will be a very small LLM)

size is the total number of elements in the tensor, used for calculations and memory allocation

Implemented Functions

tensor_create
Creates and returns a new tensor, allocating memory using malloc based on size.

tensor_free
Frees the allocated memory. After all, this is C, and we don’t want memory leaks.

tensor_print
Used only for debugging and visualization purposes. There is a test in main.c.

tensor_fill
Fills the tensor with values. When a tensor is created, memory is allocated but contains no meaningful data.

tensor_set_2d
Used to work with 2D tensors. It sets values based on row × column (i × j), just like on paper or in higher-level libraries.

tensor_get_2d
Returns a value from a 2D tensor at position (i × j).

tensor_matmul_2d
This is the core operation of the AI: matrix multiplication.
It works with compatible matrices of shape a×b * b×c, producing a matrix of shape a×c.

tensor_softmax_rows
A slightly more complex function.
Softmax is a normalization method that performs better than standard normalization. It converts each row of a matrix into values between 0 and 1, based on their magnitude.
This is essential for computing probabilities, since LLMs operate on probabilities to predict the next token.

tensor_add_inplace
Performs in-place matrix addition.