# Fundamentos Matemáticos e Teóricos para LLMs

Para construir um Modelo de Linguagem de Grande Escala (LLM) do zero, é essencial compreender os fundamentos matemáticos e teóricos que sustentam seu funcionamento. Nesta aula, revisaremos os conceitos de álgebra linear, probabilidade e estatística relevantes para LLMs, além de explorar em profundidade os mecanismos de atenção que revolucionaram o processamento de linguagem natural.

## Revisão de Álgebra Linear Relevante

A álgebra linear é a espinha dorsal dos modelos de aprendizado profundo, incluindo os LLMs. Vamos revisar os conceitos mais importantes:

### Vetores e Matrizes

**Vetores** são arranjos unidimensionais de números. Em LLMs, eles são usados para representar tokens (palavras ou subpalavras) como embeddings, estados ocultos e muito mais.

Um vetor \(\mathbf{x}\) de dimensão \(n\) pode ser representado como:

\[\mathbf{x} = [x_1, x_2, \ldots, x_n]^T\]

**Matrizes** são arranjos bidimensionais de números. Em LLMs, as matrizes representam transformações lineares, como os pesos das redes neurais.

Uma matriz \(\mathbf{W}\) de dimensão \(m \times n\) pode ser representada como:

\[\mathbf{W} = 
\begin{bmatrix} 
w_{11} & w_{12} & \cdots & w_{1n} \\
w_{21} & w_{22} & \cdots & w_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
w_{m1} & w_{m2} & \cdots & w_{mn}
\end{bmatrix}
\]

Em Python, usando NumPy:

```python
import numpy as np

# Criar um vetor
x = np.array([1, 2, 3, 4, 5])

# Criar uma matriz
W = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])
```

### Operações Fundamentais

1. **Produto Escalar (Dot Product)**: O produto escalar de dois vetores \(\mathbf{a}\) e \(\mathbf{b}\) de mesma dimensão é um escalar:

\[\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \ldots + a_n b_n\]

```python
# Produto escalar
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # ou a.dot(b) ou a @ b
print(dot_product)  # 32 = 1*4 + 2*5 + 3*6
```

2. **Multiplicação Matriz-Vetor**: Quando multiplicamos uma matriz \(\mathbf{W}\) por um vetor \(\mathbf{x}\), obtemos um novo vetor:

\[\mathbf{y} = \mathbf{W}\mathbf{x}\]

Cada elemento \(y_i\) é o produto escalar da i-ésima linha de \(\mathbf{W}\) com \(\mathbf{x}\).

```python
# Multiplicação matriz-vetor
W = np.array([[1, 2, 3], 
              [4, 5, 6]])
x = np.array([7, 8, 9])
y = W @ x
print(y)  # [50, 122]
```

3. **Multiplicação Matriz-Matriz**: O produto de duas matrizes \(\mathbf{A}\) (dimensão \(m \times n\)) e \(\mathbf{B}\) (dimensão \(n \times p\)) é uma matriz \(\mathbf{C}\) (dimensão \(m \times p\)):

\[\mathbf{C} = \mathbf{A}\mathbf{B}\]

Cada elemento \(c_{ij}\) é o produto escalar da i-ésima linha de \(\mathbf{A}\) com a j-ésima coluna de \(\mathbf{B}\).

```python
# Multiplicação matriz-matriz
A = np.array([[1, 2], 
              [3, 4], 
              [5, 6]])
B = np.array([[7, 8, 9], 
              [10, 11, 12]])
C = A @ B
print(C)
# [[27, 30, 33],
#  [61, 68, 75],
#  [95, 106, 117]]
```

### Normas e Distâncias

A **norma** de um vetor mede seu "tamanho". A norma L2 (euclidiana) é a mais comum:

\[||\mathbf{x}||_2 = \sqrt{\sum_{i=1}^{n} x_i^2}\]

A **distância euclidiana** entre dois vetores \(\mathbf{a}\) e \(\mathbf{b}\) é:

\[d(\mathbf{a}, \mathbf{b}) = ||\mathbf{a} - \mathbf{b}||_2 = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}\]

```python
# Norma L2
x = np.array([3, 4])
norm = np.linalg.norm(x)
print(norm)  # 5.0

# Distância euclidiana
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
distance = np.linalg.norm(a - b)
print(distance)  # 5.196...
```

### Decomposição em Valores Singulares (SVD)

A SVD é uma fatoração que decompõe uma matriz \(\mathbf{A}\) em três componentes:

\[\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T\]

onde \(\mathbf{U}\) e \(\mathbf{V}\) são matrizes ortogonais e \(\mathbf{\Sigma}\) é uma matriz diagonal contendo os valores singulares.

A SVD é fundamental para técnicas como PCA (Análise de Componentes Principais) e é usada em algumas otimizações de modelos de linguagem.

```python
# SVD
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, Vt = np.linalg.svd(A)
print("U:", U)
print("S:", S)
print("V^T:", Vt)

# Reconstruir A
A_reconstructed = U @ np.diag(S) @ Vt
print("A reconstruída:", A_reconstructed)
```

## Probabilidade e Estatística para Modelos de Linguagem

Os modelos de linguagem são, em sua essência, modelos probabilísticos. Vamos revisar os conceitos fundamentais:

### Probabilidade Condicional e Modelos de Linguagem

Um modelo de linguagem estima a probabilidade de uma sequência de tokens (palavras ou subpalavras). Formalmente, para uma sequência \(w_1, w_2, \ldots, w_n\), queremos calcular:

\[P(w_1, w_2, \ldots, w_n)\]

Usando a regra da cadeia de probabilidade, podemos decompor isso em:

\[P(w_1, w_2, \ldots, w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1, w_2) \cdot \ldots \cdot P(w_n|w_1, w_2, \ldots, w_{n-1})\]

Ou seja, a probabilidade de cada token depende dos tokens anteriores. Em modelos autoregressivos como GPT, o objetivo é modelar:

\[P(w_t|w_1, w_2, \ldots, w_{t-1})\]

### Entropia e Perplexidade

A **entropia** mede a incerteza associada a uma distribuição de probabilidade:

\[H(P) = -\sum_{x} P(x) \log P(x)\]

A **perplexidade** é uma medida comum para avaliar modelos de linguagem, definida como:

\[\text{Perplexidade} = 2^{H(P)} = 2^{-\frac{1}{N}\sum_{i=1}^{N} \log_2 P(w_i|w_1, \ldots, w_{i-1})}\]

Quanto menor a perplexidade, melhor o modelo.

```python
import numpy as np

# Exemplo: calcular perplexidade
# Suponha que temos as probabilidades previstas pelo modelo para cada token em uma sequência
token_probabilities = [0.2, 0.1, 0.05, 0.3, 0.15]

# Calcular log das probabilidades
log_probs = [np.log2(p) for p in token_probabilities]

# Calcular entropia média
avg_entropy = -sum(log_probs) / len(log_probs)

# Calcular perplexidade
perplexity = 2 ** avg_entropy
print(f"Perplexidade: {perplexity}")
```

### Função de Perda Cross-Entropy

A função de perda mais comum para treinar modelos de linguagem é a **cross-entropy**:

\[\text{Loss} = -\frac{1}{N}\sum_{i=1}^{N} \sum_{j=1}^{V} y_{ij} \log(p_{ij})\]

onde \(N\) é o número de exemplos, \(V\) é o tamanho do vocabulário, \(y_{ij}\) é 1 se o token \(j\) é o correto para o exemplo \(i\) (e 0 caso contrário), e \(p_{ij}\) é a probabilidade prevista pelo modelo.

Para modelos de linguagem, isso se simplifica para:

\[\text{Loss} = -\frac{1}{N}\sum_{i=1}^{N} \log(p_{i,y_i})\]

onde \(p_{i,y_i}\) é a probabilidade que o modelo atribui ao token correto \(y_i\) para o exemplo \(i\).

```python
import torch
import torch.nn.functional as F

# Exemplo: calcular cross-entropy loss
# Suponha que temos as logits (saídas não-normalizadas) do modelo
logits = torch.tensor([[2.0, 1.0, 0.1],  # Exemplo 1
                       [0.5, 2.5, 1.5]])  # Exemplo 2

# E os índices dos tokens corretos
targets = torch.tensor([0, 1])  # Classe 0 para exemplo 1, classe 1 para exemplo 2

# Calcular cross-entropy loss
loss = F.cross_entropy(logits, targets)
print(f"Loss: {loss.item()}")
```

### Amostragem e Geração de Texto

Durante a inferência, os LLMs geram texto token por token. Existem várias estratégias de amostragem:

1. **Greedy Decoding**: Sempre escolher o token com maior probabilidade.
2. **Beam Search**: Manter as \(k\) sequências mais prováveis a cada passo.
3. **Sampling com Temperatura**: Amostrar da distribuição de probabilidade, ajustada por um parâmetro de temperatura \(T\):

\[P(w_i|w_{<i}) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}\]

onde \(z_i\) são os logits do modelo e \(T\) controla a "aleatoriedade" da amostragem.

4. **Top-k Sampling**: Amostrar apenas entre os \(k\) tokens mais prováveis.
5. **Nucleus (Top-p) Sampling**: Amostrar do menor conjunto de tokens cuja probabilidade cumulativa excede \(p\).

```python
import torch
import torch.nn.functional as F
import numpy as np

# Exemplo: diferentes estratégias de amostragem
logits = torch.tensor([2.0, 1.0, 0.1, 0.5, 0.2])  # Logits para um passo de geração

# 1. Greedy decoding
greedy_idx = torch.argmax(logits).item()
print(f"Greedy: {greedy_idx}")

# 2. Sampling com temperatura
temperature = 0.8
probs = F.softmax(logits / temperature, dim=0).numpy()
sampled_idx = np.random.choice(len(probs), p=probs)
print(f"Temperature sampling: {sampled_idx}")

# 3. Top-k sampling
k = 2
top_k_logits, top_k_indices = torch.topk(logits, k)
top_k_probs = F.softmax(top_k_logits, dim=0).numpy()
sampled_idx_local = np.random.choice(len(top_k_probs), p=top_k_probs)
sampled_idx = top_k_indices[sampled_idx_local].item()
print(f"Top-k sampling: {sampled_idx}")
```

## Conceitos de Atenção e Auto-Atenção

O mecanismo de atenção é o componente central da arquitetura Transformer, que revolucionou o processamento de linguagem natural. Vamos explorar como ele funciona:

### Intuição por trás da Atenção

O mecanismo de atenção permite que um modelo "foque" em diferentes partes da entrada ao gerar cada parte da saída. A ideia fundamental é:

1. Para cada posição de saída, calcular um "score de atenção" para cada posição de entrada
2. Normalizar esses scores para obter pesos de atenção
3. Calcular uma média ponderada das representações de entrada usando esses pesos

Isso permite que o modelo estabeleça conexões diretas entre tokens distantes, superando a limitação das RNNs em capturar dependências de longo alcance.

### Atenção Scaled Dot-Product

A forma mais comum de atenção usada em Transformers é a **atenção scaled dot-product**:

1. Transformar cada vetor de entrada em três vetores: **query** (Q), **key** (K) e **value** (V)
2. Calcular os scores de atenção como o produto escalar entre queries e keys, escalado por um fator
3. Aplicar softmax para obter pesos de atenção
4. Calcular a saída como a média ponderada dos values

Matematicamente:

\[\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\]

onde \(d_k\) é a dimensão dos vetores de key.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Implementação da atenção scaled dot-product
    
    Args:
        query: tensor de shape (..., seq_len_q, d_k)
        key: tensor de shape (..., seq_len_k, d_k)
        value: tensor de shape (..., seq_len_k, d_v)
        mask: tensor opcional para mascarar posições inválidas
        
    Returns:
        output: tensor de shape (..., seq_len_q, d_v)
        attention_weights: tensor de shape (..., seq_len_q, seq_len_k)
    """
    # Calcular os scores de atenção
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    
    # Escalar os scores
    d_k = query.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Aplicar máscara se fornecida
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # Normalizar os scores com softmax
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    
    # Calcular a saída
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

### Atenção Multi-Cabeça (Multi-Head Attention)

Em vez de realizar uma única operação de atenção, os Transformers usam **atenção multi-cabeça**, que consiste em:

1. Projetar Q, K e V em múltiplos subespaços (cabeças)
2. Aplicar atenção scaled dot-product em cada cabeça independentemente
3. Concatenar os resultados e projetar de volta para a dimensão original

Isso permite que o modelo capture diferentes tipos de relações entre tokens simultaneamente.

\[\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O\]

onde:

\[\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)\]

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        """
        Divide o último dimensão em (num_heads, depth)
        e transpõe o resultado para (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Projeções lineares
        q = self.wq(query)  # (batch_size, seq_len_q, d_model)
        k = self.wk(key)    # (batch_size, seq_len_k, d_model)
        v = self.wv(value)  # (batch_size, seq_len_v, d_model)
        
        # Dividir em cabeças
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # Atenção scaled dot-product
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        # scaled_attention shape: (batch_size, num_heads, seq_len_q, depth)
        
        # Concatenar cabeças
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model)
        
        # Projeção final
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights
```

### Auto-Atenção (Self-Attention)

A **auto-atenção** é um caso especial de atenção onde query, key e value vêm da mesma fonte. Isso permite que cada token na sequência atenda a todos os outros tokens, capturando relações dentro da sequência.

Em um Transformer, a auto-atenção é aplicada separadamente no encoder (onde cada token pode atender a todos os outros tokens) e no decoder (onde cada token só pode atender a tokens anteriores, usando uma máscara).

### Atenção Mascarada

No decoder do Transformer, usamos **atenção mascarada** para garantir que a previsão de um token só dependa dos tokens anteriores (preservando a natureza autoregressiva do modelo):

```python
# Exemplo: criar uma máscara look-ahead para atenção mascarada
def create_look_ahead_mask(size):
    """
    Máscara para ocultar tokens futuros (à direita) durante o treinamento
    """
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    return mask == 1  # True para posições que devem ser mascaradas
```

### Embeddings Posicionais

Como a atenção não tem noção inerente de ordem, os Transformers adicionam **embeddings posicionais** às representações de tokens para injetar informação sobre a posição de cada token na sequência.

Os embeddings posicionais originais usam funções seno e cosseno:

\[PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)\]
\[PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)\]

onde \(pos\) é a posição e \(i\) é a dimensão.

```python
import torch
import numpy as np

def get_positional_encoding(max_seq_len, d_model):
    """
    Calcula embeddings posicionais
    
    Args:
        max_seq_len: comprimento máximo da sequência
        d_model: dimensão do modelo
        
    Returns:
        positional_encoding: tensor de shape (max_seq_len, d_model)
    """
    positional_encoding = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    
    return positional_encoding
```

## Aplicação Prática: Implementando um Mecanismo de Atenção Simples

Vamos concluir esta aula com uma implementação prática de um mecanismo de atenção simples, que servirá como base para nosso LLM:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super(SimpleAttention, self).__init__()
        self.d_model = d_model
        
        # Projeções lineares para Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Projeção de saída
        self.out_linear = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: tensor de entrada de shape (batch_size, seq_len, d_model)
            mask: tensor opcional para mascarar posições inválidas
            
        Returns:
            output: tensor de saída de shape (batch_size, seq_len, d_model)
            attention_weights: pesos de atenção
        """
        batch_size, seq_len, _ = x.size()
        
        # Projeções lineares
        q = self.q_linear(x)  # (batch_size, seq_len, d_model)
        k = self.k_linear(x)  # (batch_size, seq_len, d_model)
        v = self.v_linear(x)  # (batch_size, seq_len, d_model)
        
        # Calcular scores de atenção
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_model)  # (batch_size, seq_len, seq_len)
        
        # Aplicar máscara se fornecida
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Normalizar scores com softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Aplicar atenção aos values
        output = torch.matmul(attention_weights, v)  # (batch_size, seq_len, d_model)
        
        # Projeção final
        output = self.out_linear(output)  # (batch_size, seq_len, d_model)
        
        return output, attention_weights

# Exemplo de uso
def test_attention():
    # Parâmetros
    batch_size = 2
    seq_len = 5
    d_model = 8
    
    # Criar modelo de atenção
    attention = SimpleAttention(d_model)
    
    # Criar entrada aleatória
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Criar máscara look-ahead (para atenção causal)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, seq_len)
    
    # Forward pass
    output, attention_weights = attention(x, mask)
    
    print(f"Entrada shape: {x.shape}")
    print(f"Saída shape: {output.shape}")
    print(f"Pesos de atenção shape: {attention_weights.shape}")
    
    # Visualizar pesos de atenção para o primeiro exemplo no batch
    plt.figure(figsize=(8, 6))
    plt.matshow(attention_weights[0].detach().numpy(), cmap='viridis')
    plt.title("Pesos de Atenção")
    plt.xlabel("Posição da Key")
    plt.ylabel("Posição da Query")
    plt.colorbar()
    plt.savefig("attention_weights.png")
    plt.close()
    
    print("Visualização dos pesos de atenção salva como 'attention_weights.png'")

# Executar o teste
if __name__ == "__main__":
    test_attention()
```

## Conclusão

Nesta aula, revisamos os fundamentos matemáticos e teóricos essenciais para compreender e implementar LLMs. Exploramos conceitos de álgebra linear, probabilidade e estatística, e nos aprofundamos no mecanismo de atenção que é o coração da arquitetura Transformer.

Estes conceitos fornecem a base teórica necessária para implementarmos nosso próprio LLM do zero nas próximas aulas. No próximo módulo, começaremos a preparar nosso ambiente de desenvolvimento e a implementar os primeiros componentes do nosso modelo.

## Referências e Leituras Adicionais

1. Vaswani, A., et al. (2017). "Attention Is All You Need". Neural Information Processing Systems.
2. Alammar, J. "The Illustrated Transformer". http://jalammar.github.io/illustrated-transformer/
3. Jurafsky, D. & Martin, J. H. "Speech and Language Processing". https://web.stanford.edu/~jurafsky/slp3/
4. Deng, L. & Liu, Y. (2018). "Deep Learning in Natural Language Processing".
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning". MIT Press.
