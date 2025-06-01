# Componentes Fundamentais do Transformer

Neste módulo, vamos explorar os componentes fundamentais da arquitetura Transformer, que revolucionou o processamento de linguagem natural e serve como base para os LLMs modernos. Vamos implementar cada componente do zero, entendendo seu funcionamento interno e como eles se combinam para formar um modelo completo.

## Embeddings: A Base da Representação

Antes que um modelo possa processar texto, precisamos converter tokens em representações vetoriais densas chamadas embeddings. Vamos explorar dois tipos essenciais de embeddings usados em Transformers.

### Embeddings de Tokens

Os embeddings de tokens mapeiam cada token do vocabulário para um vetor de dimensão fixa. Estes vetores capturam características semânticas e sintáticas dos tokens.

```python
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        Inicializa a camada de embedding de tokens.
        
        Args:
            vocab_size: Tamanho do vocabulário
            embedding_dim: Dimensão do embedding
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        
        # Inicialização dos pesos (importante para convergência)
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim ** -0.5)
    
    def forward(self, x):
        """
        Args:
            x: Tensor de índices de tokens de shape (batch_size, seq_len)
            
        Returns:
            Tensor de embeddings de shape (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(x) * math.sqrt(self.embedding_dim)
```

A multiplicação por `sqrt(embedding_dim)` é uma técnica de escala que ajuda a estabilizar o treinamento, conforme proposto no artigo original do Transformer.

### Embeddings Posicionais

Como a arquitetura Transformer processa todos os tokens em paralelo (sem recorrência), precisamos injetar informação sobre a posição de cada token na sequência. Os embeddings posicionais fazem exatamente isso.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=5000, dropout=0.1):
        """
        Inicializa a codificação posicional.
        
        Args:
            embedding_dim: Dimensão do embedding
            max_seq_length: Comprimento máximo de sequência suportado
            dropout: Taxa de dropout
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Criar matriz de codificação posicional
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        # Aplicar funções seno e cosseno
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Adicionar dimensão de batch e registrar como buffer (não é um parâmetro)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor de embeddings de tokens de shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            Tensor com embeddings posicionais adicionados
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

Os embeddings posicionais usam funções seno e cosseno de diferentes frequências para cada dimensão. Isso permite que o modelo aprenda a atender a posições relativas, mesmo para sequências mais longas do que as vistas durante o treinamento.

Vamos visualizar como os embeddings posicionais se parecem:

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_positional_encoding(pe, dim=64, seq_len=100):
    """
    Visualiza os embeddings posicionais.
    
    Args:
        pe: Tensor de embeddings posicionais
        dim: Número de dimensões a mostrar
        seq_len: Comprimento da sequência a mostrar
    """
    plt.figure(figsize=(10, 6))
    pe_slice = pe[0, :seq_len, :dim].numpy()
    plt.imshow(pe_slice, cmap='viridis', aspect='auto')
    plt.xlabel('Dimensão do Embedding')
    plt.ylabel('Posição na Sequência')
    plt.colorbar(label='Valor')
    plt.title('Visualização dos Embeddings Posicionais')
    plt.savefig('positional_encodings.png')
    plt.close()

# Exemplo de uso
# model_dim = 64
# pos_encoder = PositionalEncoding(model_dim, max_seq_length=200)
# pos_encodings = pos_encoder.pe.detach()
# plot_positional_encoding(pos_encodings)
```

### Embeddings Combinados

Na prática, combinamos embeddings de tokens e posicionais para obter a representação completa de entrada:

```python
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length=5000, dropout=0.1):
        """
        Combina embeddings de tokens e posicionais.
        
        Args:
            vocab_size: Tamanho do vocabulário
            embedding_dim: Dimensão do embedding
            max_seq_length: Comprimento máximo de sequência
            dropout: Taxa de dropout
        """
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length, dropout)
    
    def forward(self, x):
        """
        Args:
            x: Tensor de índices de tokens de shape (batch_size, seq_len)
            
        Returns:
            Tensor de embeddings combinados de shape (batch_size, seq_len, embedding_dim)
        """
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        return x
```

## Mecanismo de Atenção Multi-Cabeça

O coração do Transformer é o mecanismo de atenção, que permite que o modelo foque em diferentes partes da entrada ao gerar cada parte da saída.

### Atenção Scaled Dot-Product

Primeiro, implementamos a atenção scaled dot-product básica:

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Calcula a atenção scaled dot-product.
    
    Args:
        query: Tensor de shape (..., seq_len_q, d_k)
        key: Tensor de shape (..., seq_len_k, d_k)
        value: Tensor de shape (..., seq_len_k, d_v)
        mask: Tensor opcional para mascarar posições inválidas
        
    Returns:
        output: Tensor de shape (..., seq_len_q, d_v)
        attention_weights: Tensor de shape (..., seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)
    
    # Produto escalar de query e key
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Aplicar máscara (se fornecida)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Aplicar softmax para obter pesos de atenção
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    
    # Aplicar pesos aos values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

### Atenção Multi-Cabeça

A atenção multi-cabeça permite que o modelo atenda a diferentes representações subspace simultaneamente:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Inicializa o módulo de atenção multi-cabeça.
        
        Args:
            d_model: Dimensão do modelo
            num_heads: Número de cabeças de atenção
            dropout: Taxa de dropout
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimensão por cabeça
        
        # Projeções lineares para Q, K, V
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # Projeção de saída
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        """
        Divide o último dimensão em (num_heads, d_k)
        e transpõe para (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_length = x.size(0), x.size(1)
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)
    
    def combine_heads(self, x):
        """
        Combina as cabeças de atenção.
        Transpõe de (batch_size, num_heads, seq_len, d_k)
        para (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_length, _ = x.size()
        x = x.permute(0, 2, 1, 3)
        return x.contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Tensor de shape (batch_size, seq_len_q, d_model)
            key: Tensor de shape (batch_size, seq_len_k, d_model)
            value: Tensor de shape (batch_size, seq_len_v, d_model)
            mask: Tensor opcional para mascarar posições inválidas
            
        Returns:
            output: Tensor de shape (batch_size, seq_len_q, d_model)
            attention_weights: Tensor de shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)
        
        # Projeções lineares e divisão em cabeças
        q = self.split_heads(self.wq(query))  # (batch_size, num_heads, seq_len_q, d_k)
        k = self.split_heads(self.wk(key))    # (batch_size, num_heads, seq_len_k, d_k)
        v = self.split_heads(self.wv(value))  # (batch_size, num_heads, seq_len_v, d_k)
        
        # Atenção scaled dot-product para cada cabeça
        output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # Combinar cabeças
        output = self.combine_heads(output)  # (batch_size, seq_len_q, d_model)
        
        # Projeção final
        output = self.wo(output)
        output = self.dropout(output)
        
        return output, attention_weights
```

## Feed-Forward Networks

Após a atenção, cada posição na sequência passa por uma rede feed-forward idêntica:

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Inicializa a rede feed-forward.
        
        Args:
            d_model: Dimensão do modelo
            d_ff: Dimensão interna da feed-forward
            dropout: Taxa de dropout
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU é comum em modelos modernos como GPT
    
    def forward(self, x):
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)
```

## Normalização de Camada

A normalização de camada é crucial para estabilizar o treinamento de Transformers:

```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        Inicializa a normalização de camada.
        
        Args:
            features: Número de features
            eps: Valor epsilon para estabilidade numérica
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, features)
            
        Returns:
            Tensor normalizado de mesma shape
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

## Montagem do Encoder

Agora, vamos combinar esses componentes para formar um bloco de encoder completo:

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Inicializa uma camada do encoder.
        
        Args:
            d_model: Dimensão do modelo
            num_heads: Número de cabeças de atenção
            d_ff: Dimensão interna da feed-forward
            dropout: Taxa de dropout
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, d_model)
            mask: Tensor opcional para mascarar posições inválidas
            
        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """
        # Auto-atenção com conexão residual e normalização
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward com conexão residual e normalização
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length=5000, dropout=0.1):
        """
        Inicializa o encoder completo.
        
        Args:
            vocab_size: Tamanho do vocabulário
            d_model: Dimensão do modelo
            num_heads: Número de cabeças de atenção
            d_ff: Dimensão interna da feed-forward
            num_layers: Número de camadas do encoder
            max_seq_length: Comprimento máximo de sequência
            dropout: Taxa de dropout
        """
        super(Encoder, self).__init__()
        
        self.embedding_layer = EmbeddingLayer(vocab_size, d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor de índices de tokens de shape (batch_size, seq_len)
            mask: Tensor opcional para mascarar posições inválidas
            
        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """
        # Aplicar embeddings
        x = self.embedding_layer(x)
        
        # Passar por cada camada do encoder
        for layer in self.layers:
            x = layer(x, mask)
        
        # Normalização final
        return self.norm(x)
```

## Implementação do Decoder

O decoder é semelhante ao encoder, mas com uma camada adicional de atenção cruzada:

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Inicializa uma camada do decoder.
        
        Args:
            d_model: Dimensão do modelo
            num_heads: Número de cabeças de atenção
            d_ff: Dimensão interna da feed-forward
            dropout: Taxa de dropout
        """
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            x: Tensor de shape (batch_size, seq_len, d_model)
            enc_output: Saída do encoder de shape (batch_size, enc_seq_len, d_model)
            look_ahead_mask: Máscara para ocultar tokens futuros
            padding_mask: Máscara para ocultar tokens de padding
            
        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """
        # Auto-atenção mascarada
        attn1_output, _ = self.self_attn(x, x, x, look_ahead_mask)
        x = self.norm1(x + self.dropout(attn1_output))
        
        # Atenção cruzada com saída do encoder
        attn2_output, _ = self.cross_attn(x, enc_output, enc_output, padding_mask)
        x = self.norm2(x + self.dropout(attn2_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length=5000, dropout=0.1):
        """
        Inicializa o decoder completo.
        
        Args:
            vocab_size: Tamanho do vocabulário
            d_model: Dimensão do modelo
            num_heads: Número de cabeças de atenção
            d_ff: Dimensão interna da feed-forward
            num_layers: Número de camadas do decoder
            max_seq_length: Comprimento máximo de sequência
            dropout: Taxa de dropout
        """
        super(Decoder, self).__init__()
        
        self.embedding_layer = EmbeddingLayer(vocab_size, d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            x: Tensor de índices de tokens de shape (batch_size, seq_len)
            enc_output: Saída do encoder de shape (batch_size, enc_seq_len, d_model)
            look_ahead_mask: Máscara para ocultar tokens futuros
            padding_mask: Máscara para ocultar tokens de padding
            
        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """
        # Aplicar embeddings
        x = self.embedding_layer(x)
        
        # Passar por cada camada do decoder
        for layer in self.layers:
            x = layer(x, enc_output, look_ahead_mask, padding_mask)
        
        # Normalização final
        return self.norm(x)
```

## Atenção Mascarada

No decoder, precisamos garantir que cada posição só possa atender a posições anteriores (para preservar a natureza autoregressiva do modelo). Para isso, usamos uma máscara "look-ahead":

```python
def create_masks(src, tgt=None):
    """
    Cria máscaras para encoder e decoder.
    
    Args:
        src: Tensor de índices de tokens de entrada de shape (batch_size, src_seq_len)
        tgt: Tensor opcional de índices de tokens alvo de shape (batch_size, tgt_seq_len)
        
    Returns:
        Dicionário de máscaras
    """
    # Máscara de padding para src (1 para tokens reais, 0 para padding)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_len)
    
    masks = {'src_mask': src_mask}
    
    if tgt is not None:
        # Máscara de padding para tgt
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_seq_len)
        
        # Máscara look-ahead para tgt
        tgt_seq_len = tgt.size(1)
        look_ahead_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len)), diagonal=1).bool()
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_seq_len, tgt_seq_len)
        
        # Combinar máscaras de padding e look-ahead
        combined_mask = tgt_padding_mask & ~look_ahead_mask
        
        masks.update({
            'tgt_mask': combined_mask,
            'memory_mask': tgt_padding_mask
        })
    
    return masks
```

## Montagem do Transformer Completo

Finalmente, vamos juntar todos os componentes para formar um modelo Transformer completo:

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, 
                 num_encoder_layers=6, num_decoder_layers=6, max_seq_length=5000, dropout=0.1):
        """
        Inicializa o modelo Transformer completo.
        
        Args:
            src_vocab_size: Tamanho do vocabulário de entrada
            tgt_vocab_size: Tamanho do vocabulário de saída
            d_model: Dimensão do modelo
            num_heads: Número de cabeças de atenção
            d_ff: Dimensão interna da feed-forward
            num_encoder_layers: Número de camadas do encoder
            num_decoder_layers: Número de camadas do decoder
            max_seq_length: Comprimento máximo de sequência
            dropout: Taxa de dropout
        """
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, d_ff, 
            num_encoder_layers, max_seq_length, dropout
        )
        
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads, d_ff, 
            num_decoder_layers, max_seq_length, dropout
        )
        
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt):
        """
        Args:
            src: Tensor de índices de tokens de entrada de shape (batch_size, src_seq_len)
            tgt: Tensor de índices de tokens alvo de shape (batch_size, tgt_seq_len)
            
        Returns:
            Tensor de logits de shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Criar máscaras
        masks = create_masks(src, tgt)
        
        # Codificar a entrada
        enc_output = self.encoder(src, masks['src_mask'])
        
        # Decodificar
        dec_output = self.decoder(
            tgt, enc_output, 
            masks.get('tgt_mask'), masks.get('memory_mask')
        )
        
        # Projeção final para obter logits
        logits = self.final_layer(dec_output)
        
        return logits
```

## Modelo de Linguagem Causal (Apenas Decoder)

Para LLMs como GPT, geralmente usamos apenas a parte decoder do Transformer, em uma configuração autoregressiva:

```python
class CausalLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, d_ff=3072, 
                 num_layers=12, max_seq_length=1024, dropout=0.1):
        """
        Inicializa um modelo de linguagem causal (apenas decoder).
        
        Args:
            vocab_size: Tamanho do vocabulário
            d_model: Dimensão do modelo
            num_heads: Número de cabeças de atenção
            d_ff: Dimensão interna da feed-forward
            num_layers: Número de camadas
            max_seq_length: Comprimento máximo de sequência
            dropout: Taxa de dropout
        """
        super(CausalLanguageModel, self).__init__()
        
        self.embedding_layer = EmbeddingLayer(vocab_size, d_model, max_seq_length, dropout)
        
        # Camadas do decoder (sem atenção cruzada)
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Compartilhar pesos entre embedding e camada de saída (técnica comum)
        self.lm_head.weight = self.embedding_layer.token_embedding.embedding.weight
    
    def forward(self, x):
        """
        Args:
            x: Tensor de índices de tokens de shape (batch_size, seq_len)
            
        Returns:
            Tensor de logits de shape (batch_size, seq_len, vocab_size)
        """
        # Criar máscara causal
        seq_len = x.size(1)
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Aplicar embeddings
        x = self.embedding_layer(x)
        
        # Passar por cada camada
        for layer in self.layers:
            x = layer(x, mask)
        
        # Normalização final
        x = self.norm(x)
        
        # Projeção para logits
        logits = self.lm_head(x)
        
        return logits

class DecoderOnlyLayer(nn.Module):
    """
    Camada de decoder sem atenção cruzada, usada em modelos como GPT.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderOnlyLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Auto-atenção mascarada
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## Exemplo de Uso

Vamos ver como usar nosso modelo em um exemplo simples:

```python
def test_causal_lm():
    """
    Testa o modelo de linguagem causal.
    """
    # Parâmetros do modelo
    vocab_size = 10000
    d_model = 256
    num_heads = 4
    d_ff = 1024
    num_layers = 4
    max_seq_length = 128
    
    # Criar modelo
    model = CausalLanguageModel(
        vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length
    )
    
    # Dados de exemplo
    batch_size = 2
    seq_len = 10
    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Verificar se as dimensões estão corretas
    assert logits.shape == (batch_size, seq_len, vocab_size), "Dimensões incorretas!"
    print("Teste bem-sucedido!")

# Executar teste
# test_causal_lm()
```

## Conclusão

Nesta aula, implementamos os componentes fundamentais da arquitetura Transformer, incluindo:

1. Embeddings de tokens e posicionais
2. Mecanismo de atenção multi-cabeça
3. Redes feed-forward
4. Normalização de camada
5. Blocos de encoder e decoder
6. Modelo Transformer completo
7. Modelo de linguagem causal (apenas decoder)

Estes componentes formam a base dos LLMs modernos como GPT, BERT e T5. Na próxima aula, vamos explorar como otimizar esses modelos para funcionar em hardware com limitações de memória.

## Exercícios Práticos

1. Implemente uma versão simplificada do modelo CausalLanguageModel com apenas 2 camadas e teste-o com uma sequência curta.
2. Modifique o código para adicionar residual connections antes da normalização (pré-normalização), uma técnica usada em modelos como GPT-2.
3. Implemente uma função para gerar texto autoregressivamente usando o modelo CausalLanguageModel.
4. Experimente com diferentes funções de ativação (ReLU, Swish) na camada feed-forward e compare com GELU.
5. Visualize os pesos de atenção gerados pelo modelo para uma sequência de exemplo.
