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

Além dos tensores, que é a estrutura de dados que utilizaremos, precisaremos de um "dicionário" para o LLM. Este dicionário é o Tokenizer. O Tokenizer é uma espécie de tradutor de palavras para números (mas não tão simples assim). Imagine que você tem a palavra ferver. Ferver é um verbo, e pode ser conjugado como ferveu, fervido, fervor, fervendo, e nele há um padrão... o "ferv". "Ferv" sempre antecipa as conjugações, então invés de criarmos um número e uma posição no dicionário para cada conjugação, podemos simplesmente criar uma para ferv e uma para eu, por exemplo. Assim, o modelo sabe que "ferv" representa o verbo, e "eu" representa no passado (em verbos de segunda conjugação) e isso armazena informação implícita nas palavras. Criaremos nosso tokenizer antes da rede neural.

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

# Dia 19/01/2025

Agora, vamos para outra parte essencial do nosso LLM. Temos a estrutura de dados definida (os tensores) e agora teremos a construção do vocabulário do nosso LLM. 

ATENÇÃO: USAREMOS O TOKENIZER DO ANDREJ! https://github.com/karpathy/llama2.c/blob/master/tokenizer.bin

Dentro de tokenizer.h, temos o struct:

typedef struct {
    char **vocab;          
    float *vocab_scores;  
    int vocab_size;        
    int max_token_length;  
    unsigned char *byte_pieces;
} Tokenizer;

Este é o nosso tokenizer, que ficará salvo em RAM, enquanto o programa rodar. Neste exemplo, ele é pequeno, então não há problemas, a menos que você escolha um dicionário extremamente grande. O que usaremos será pequeno. 

Dentro do struct armazenamos informações importantes sobre o dicionário:

**vocab é uma matriz de strings, logo, é o conteúdo do nosso dicionário. Lá estará armazenado nosso vocabulário. 

*vocab_scores é o SCORE dos bytes que o nosso modelo usará para saber qual combinação é ideal para formar uma palavra. Isso é um pouco complexo, mas, mais acima, citei sobre como ferv + eu forma a palavra ferveu. O modelo não recebe palavras inteiras de uma vez, mas letras, e vai montando as palavras mais prováveis de acordo com o que viu no treinamento, por exemplo: f,e,r,v,e,u, e monta: fe,r,v,e,u, porque viu que "fe" tem um score maior que "f" e "e" juntos, em seguida, vai seguindo, até montar ferveu.

vocab_size é o tamanho do nosso vocabulário, isso porque tudo no C é bem especificado. Precisamos de tamanho para saber o quanto de memória precisamos alocar.

*byte_pices é simplesmente... um byte puro! Ele serve para caso um caractere estranho (como emojis, letras diferentes, etc) entrem no modelo, ele possa simplesmente trocar este caractere por um byte bruto e não quebrar.

Para as funções:

tokenizer_init é a função que lerá nosso arquivo de dicionário e inicializará ele na RAM. É uma função simples, que abre o arquivo e monta o nosso tokenizer. Não tem retorno, pois como recebe o endereço de memória do tokenizer, ele já arruma ele ali mesmo.

Coloquei str_lookup no tokenizer.h apenas para testes, ficará lá até a proxima atualização. É a função de busca no nosso modelo. O tokenizer_encode pegará a string e procurará qual o número dela dentro do dicionário. Usamos uma busca linear simples pois não há hash table nativa no C. Provavelmente adicionarei uma construção de hash nas próximas atualizações, mas para funcionar o tokenizer, vamos assim mesmo. 

tokenizer_free é o limpa memória do tokenizer, para não haver erros de memória.

tokenizer_encode é o que transformará a string em seu token numérico. Perceba que o algoritmo tem comportamento guloso: ele utiliza um while com um valor de score extremamente negativo para começar a contagem da melhor combinação de chars e fica com a que tiver o score mais alto. Com isso, ele saberá qual a melhor palavra montar e encontrará o número para ela.

tokenizer_decode é o processo reverso e mais simples. Ele já tem o número, agora, temos que devolver!

Com isso, temos o nosso sistema pronto: o leitor do dicionário, o tradutor que transforma chars em números e depois números em chars e o "coletor de lixo" que limpa a memória em seguida. Nos próximos passos, vamos começar a implementar cálculos necessários ao LLM, a rede neural e como integrar isso tudo.

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

# Day 19/01/2025

Now, let’s move on to another essential part of our LLM.
We already have the data structures defined (the tensors), and now we will build the vocabulary of our LLM.

Inside tokenizer.h, we have the following struct:

typedef struct {
    char **vocab;          
    float *vocab_scores;  
    int vocab_size;        
    int max_token_length;  
    unsigned char *byte_pieces;
} Tokenizer;


This is our tokenizer, which will remain stored in RAM while the program is running.
In this example, it is small, so there are no issues unless you choose an extremely large dictionary. The one we will use is intentionally small.

Stored information in the struct

Each field stores important information about the dictionary:

vocab is an array of strings, meaning it contains the actual content of our dictionary. This is where the vocabulary entries are stored.

vocab_scores stores the score of each token. These scores are used by the model to decide which byte combinations are best for forming words.
This is a bit complex, but as mentioned earlier, this is how combinations like boil + ed form the word boiled.
The model does not receive full words at once. Instead, it receives characters and progressively builds the most probable combinations based on what it learned during training. For example:
b, o, i, l, e, d
becomes
bo, i, l, e, d
because "bo" has a higher score than "b" and "o" separately. This process continues until the full word boiled is formed.

vocab_size is the size of the vocabulary. Since C requires explicit memory management, we need this value to know how much memory must be allocated.

byte_pieces is simply a raw byte fallback. It exists so that if an unknown or unusual character (such as emojis or unsupported symbols) enters the model, it can be replaced by a raw byte instead of breaking the tokenizer.

Functions overview

WE WILL USE ANDREJ'S TOKENIZER FOR THIS PROJECT

https://github.com/karpathy/llama2.c/blob/master/tokenizer.bin

tokenizer_init
This function reads the vocabulary file and initializes the tokenizer in RAM.
It does not return anything because it receives a pointer to the tokenizer struct and fills it directly in memory.

str_lookup
This function exists mainly for testing purposes and will remain in tokenizer.h until the next update.
It performs string lookup in the vocabulary.

tokenizer_encode
This function converts a string into its numeric token representation.
It uses a simple linear search, since C does not provide a native hash table. In future updates, a hash-based structure may be added for performance, but for now this approach is sufficient.

The algorithm is greedy:
it uses a while loop starting from a very negative score value and searches for the character combination with the highest score.
By doing this, it determines the best possible token at each step, builds the word incrementally, and finally finds the corresponding token ID in the vocabulary.

tokenizer_decode
This is the reverse and simpler process.
Given a token ID, it retrieves and returns the corresponding string.

tokenizer_free
This function releases all allocated memory used by the tokenizer, preventing memory leaks.

Final overview:

With this, we now have a complete system:

A dictionary loader, a translator that converts characters into numeric tokens, reverse translator that converts tokens back into text, a memory cleanup routine. In the next steps, we will begin implementing the core LLM computations, including the neural network itself and how all these components are integrated together.