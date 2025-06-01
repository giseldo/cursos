# Construindo um LLM Compacto

Neste módulo, vamos implementar um Modelo de Linguagem de Grande Escala (LLM) compacto, otimizado para funcionar em hardware com limitações de memória, como GPUs com 4-6GB ou no Google Colab. Vamos explorar técnicas de otimização que permitem treinar e executar modelos surpreendentemente poderosos mesmo com recursos computacionais limitados.

## Arquitetura do Modelo

Vamos começar definindo a arquitetura do nosso LLM compacto, que será baseada no modelo GPT (Generative Pre-trained Transformer), mas com modificações para torná-lo mais eficiente em termos de memória.

### Definição da Estrutura do Modelo

Nossa implementação será baseada na arquitetura de apenas decoder do Transformer, semelhante ao GPT-2, mas com parâmetros reduzidos:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class CompactLLM(nn.Module):
    """
    Um modelo de linguagem compacto baseado na arquitetura Transformer.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 384,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 6,
        intermediate_size: int = 1536,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1024,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        use_cache: bool = True,
    ):
        """
        Inicializa o modelo CompactLLM.
        
        Args:
            vocab_size: Tamanho do vocabulário
            hidden_size: Dimensão dos embeddings e camadas ocultas
            num_hidden_layers: Número de camadas do Transformer
            num_attention_heads: Número de cabeças de atenção
            intermediate_size: Dimensão da camada feed-forward
            hidden_dropout_prob: Taxa de dropout para camadas ocultas
            attention_probs_dropout_prob: Taxa de dropout para atenção
            max_position_embeddings: Número máximo de posições para embeddings posicionais
            initializer_range: Desvio padrão da inicialização normal
            layer_norm_eps: Epsilon para layer normalization
            pad_token_id: ID do token de padding
            use_cache: Se True, retorna estados passados para geração mais rápida
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.use_cache = use_cache
        
        # Camada de embeddings
        self.embeddings = TransformerEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
        )
        
        # Camadas do Transformer
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                attention_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Normalização final
        self.ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Camada de saída (compartilha pesos com embeddings)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Inicialização
        self._init_weights()
        
        # Compartilhar pesos entre embeddings e camada de saída
        self.tie_weights()
    
    def _init_weights(self):
        """
        Inicializa os pesos do modelo.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def tie_weights(self):
        """
        Compartilha pesos entre a camada de embeddings e a camada de saída.
        """
        self.lm_head.weight = self.embeddings.word_embeddings.weight
    
    def get_input_embeddings(self):
        """
        Retorna a camada de embeddings de palavras.
        """
        return self.embeddings.word_embeddings
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass do modelo.
        
        Args:
            input_ids: Tensor de índices de tokens de shape (batch_size, seq_len)
            attention_mask: Máscara de atenção de shape (batch_size, seq_len)
            position_ids: IDs de posição de shape (batch_size, seq_len)
            past_key_values: Valores de cache para geração autoregressiva
            use_cache: Se True, retorna estados passados para geração mais rápida
            return_dict: Se True, retorna um dicionário, caso contrário retorna uma tupla
            
        Returns:
            Logits para próximos tokens e opcionalmente past_key_values
        """
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        # Obter embeddings
        hidden_states = self.embeddings(input_ids, position_ids)
        
        # Criar máscara de atenção causal se não fornecida
        batch_size, seq_length = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        
        # Converter máscara de atenção para formato adequado (1 para tokens a serem atendidos, 0 para mascarados)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Inicializar past_key_values se não fornecido
        if past_key_values is None:
            past_key_values = tuple([None] * self.num_hidden_layers)
        
        all_hidden_states = ()
        all_self_attentions = ()
        next_decoder_cache = ()
        
        # Passar por cada camada do Transformer
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
            
            all_self_attentions = all_self_attentions + (layer_outputs[2],)
        
        # Normalização final
        hidden_states = self.ln_f(hidden_states)
        
        all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Calcular logits
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            return (logits, next_decoder_cache, all_hidden_states, all_self_attentions)
        
        return {
            "logits": logits,
            "past_key_values": next_decoder_cache if use_cache else None,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
        }
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        """
        Gera texto de forma autoregressiva.
        
        Args:
            input_ids: Tensor de índices de tokens de contexto
            max_length: Comprimento máximo da sequência gerada
            temperature: Temperatura para amostragem
            top_k: Número de tokens mais prováveis a considerar
            top_p: Probabilidade cumulativa para nucleus sampling
            repetition_penalty: Penalidade para repetição de tokens
            do_sample: Se True, amostra da distribuição, caso contrário usa greedy decoding
            pad_token_id: ID do token de padding
            eos_token_id: ID do token de fim de sequência
            
        Returns:
            Tensor de índices de tokens gerados
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        # Inicializar past_key_values
        past = None
        
        # Continuar gerando até atingir max_length
        while cur_len < max_length:
            # Preparar modelo para geração
            model_inputs = self.prepare_inputs_for_generation(input_ids, past)
            
            # Forward pass
            outputs = self(**model_inputs, use_cache=True, return_dict=True)
            next_token_logits = outputs["logits"][:, -1, :]
            past = outputs["past_key_values"]
            
            # Aplicar penalidade de repetição
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(input_ids[i].tolist()):
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty
            
            # Aplicar temperatura
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Mascarar tokens de padding
            if pad_token_id is not None:
                next_token_logits[:, pad_token_id] = -float("inf")
            
            # Amostragem
            if do_sample:
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("inf")
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = -float("inf")
                
                # Amostra da distribuição
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Adicionar token gerado à sequência
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            cur_len += 1
            
            # Verificar se atingiu token de fim
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break
        
        return input_ids
    
    def prepare_inputs_for_generation(self, input_ids, past=None):
        """
        Prepara entradas para geração autoregressiva.
        """
        # Apenas o último token para a próxima previsão se estamos usando past
        if past is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        
        return {
            "input_ids": input_ids,
            "past_key_values": past,
        }
```

### Implementação dos Componentes

Agora, vamos implementar os componentes necessários para o nosso modelo:

#### Camada de Embeddings

```python
class TransformerEmbeddings(nn.Module):
    """
    Camada de embeddings para o Transformer.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        dropout_prob: float,
        layer_norm_eps: float,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Registrar buffer para posições
        position_ids = torch.arange(max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        """
        Forward pass da camada de embeddings.
        
        Args:
            input_ids: Tensor de índices de tokens
            position_ids: Tensor opcional de índices de posição
            
        Returns:
            Embeddings combinados
        """
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        # Obter embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Somar embeddings
        embeddings = inputs_embeds + position_embeds
        
        # Normalizar e aplicar dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
```

#### Camada de Atenção

```python
class MultiHeadAttention(nn.Module):
    """
    Implementação otimizada de atenção multi-cabeça.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout_prob: float,
        is_causal: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.is_causal = is_causal
        
        # Verificar se hidden_size é divisível por num_attention_heads
        assert hidden_size % num_attention_heads == 0, \
            f"hidden_size {hidden_size} não é divisível por num_attention_heads {num_attention_heads}"
        
        # Projeções lineares para Q, K, V
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Projeção de saída
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(attention_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reorganiza o tensor para separar as cabeças de atenção.
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], torch.Tensor]:
        """
        Forward pass da camada de atenção multi-cabeça.
        
        Args:
            hidden_states: Tensor de estados ocultos
            attention_mask: Máscara de atenção opcional
            past_key_value: Cache opcional de chaves e valores passados
            use_cache: Se True, retorna chaves e valores para uso futuro
            
        Returns:
            Tupla contendo:
                - estados de saída
                - cache de chaves e valores (se use_cache=True)
                - pesos de atenção
        """
        # Projeções lineares
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        # Se temos past_key_value, usamos ele em vez de calcular novamente
        if past_key_value is not None:
            key_layer, value_layer = past_key_value
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Cache para geração autoregressiva
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None
        
        # Calcular scores de atenção
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Aplicar máscara de atenção
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Aplicar máscara causal se necessário
        if self.is_causal and attention_mask is None:
            seq_len = hidden_states.size(1)
            causal_mask = torch.triu(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden_states.device),
                diagonal=1
            )
            attention_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -10000.0)
        
        # Normalizar scores com softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Aplicar atenção aos values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Combinar cabeças
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Projeção final
        output = self.output(context_layer)
        
        return output, present, attention_probs
```

#### Camada Feed-Forward

```python
class FeedForward(nn.Module):
    """
    Rede feed-forward do Transformer.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_prob: float,
    ):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da camada feed-forward.
        
        Args:
            hidden_states: Tensor de estados ocultos
            
        Returns:
            Tensor transformado
        """
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states
```

#### Camada Completa do Transformer

```python
class TransformerLayer(nn.Module):
    """
    Camada completa do Transformer (atenção + feed-forward).
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        attention_dropout_prob: float,
        hidden_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        
        # Atenção multi-cabeça
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout_prob=attention_dropout_prob,
            is_causal=True,
        )
        
        # Feed-forward
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_prob=hidden_dropout_prob,
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], torch.Tensor]:
        """
        Forward pass da camada do Transformer.
        
        Args:
            hidden_states: Tensor de estados ocultos
            attention_mask: Máscara de atenção opcional
            past_key_value: Cache opcional de chaves e valores passados
            use_cache: Se True, retorna chaves e valores para uso futuro
            
        Returns:
            Tupla contendo:
                - estados de saída
                - cache de chaves e valores (se use_cache=True)
                - pesos de atenção
        """
        # Atenção com conexão residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output, present, attention_probs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropout(attn_output)
        
        # Feed-forward com conexão residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + ff_output
        
        return hidden_states, present, attention_probs
```

## Técnicas de Otimização de Memória

Para executar nosso LLM em hardware com limitações de memória, precisamos implementar várias técnicas de otimização.

### Quantização de Pesos

A quantização reduz a precisão dos parâmetros do modelo, economizando memória:

```python
def quantize_model(model, quantization_bit=8):
    """
    Quantiza os pesos do modelo para economizar memória.
    
    Args:
        model: Modelo a ser quantizado
        quantization_bit: Número de bits para quantização (8 ou 4)
        
    Returns:
        Modelo quantizado
    """
    import torch.quantization as quantization
    
    # Configurar esquema de quantização
    if quantization_bit == 8:
        # Quantização de 8 bits
        model.qconfig = quantization.get_default_qconfig('fbgemm')
    elif quantization_bit == 4:
        # Quantização de 4 bits (experimental)
        # Nota: PyTorch não suporta nativamente quantização de 4 bits
        # Esta é uma implementação simplificada
        raise NotImplementedError("Quantização de 4 bits não implementada nativamente no PyTorch")
    else:
        raise ValueError(f"Bits de quantização não suportados: {quantization_bit}")
    
    # Preparar para quantização
    model_prepared = quantization.prepare(model)
    
    # Calibrar o modelo (normalmente feito com dados de calibração)
    # Aqui, estamos pulando a calibração para simplificar
    
    # Converter para modelo quantizado
    model_quantized = quantization.convert(model_prepared)
    
    return model_quantized
```

### Gradient Checkpointing

O gradient checkpointing economiza memória durante o treinamento recalculando ativações intermediárias durante o backward pass:

```python
def enable_gradient_checkpointing(model):
    """
    Habilita gradient checkpointing para economizar memória durante o treinamento.
    
    Args:
        model: Modelo a ser modificado
        
    Returns:
        Modelo com gradient checkpointing habilitado
    """
    # Verificar se o modelo é uma instância de CompactLLM
    if not isinstance(model, CompactLLM):
        raise ValueError("O modelo deve ser uma instância de CompactLLM")
    
    # Definir função de checkpoint para as camadas do Transformer
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward
    
    # Aplicar gradient checkpointing a cada camada
    for layer in model.layers:
        layer.gradient_checkpointing = True
        
        # Substituir o método forward original
        original_forward = layer.forward
        
        def checkpointed_forward(self, *args, **kwargs):
            if getattr(self, "gradient_checkpointing", False) and self.training:
                return torch.utils.checkpoint.checkpoint(
                    create_custom_forward(original_forward.__get__(self, type(self))),
                    *args,
                    **kwargs
                )
            else:
                return original_forward.__get__(self, type(self))(*args, **kwargs)
        
        layer.forward = types.MethodType(checkpointed_forward, layer)
    
    return model
```

### Offloading para CPU

O offloading move temporariamente partes do modelo para a CPU quando não estão em uso:

```python
class CPUOffloadWrapper(nn.Module):
    """
    Wrapper para offloading de camadas para CPU.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.device = next(module.parameters()).device
    
    def forward(self, *args, **kwargs):
        # Mover módulo para GPU
        self.module.to(self.device)
        
        # Forward pass
        result = self.module(*args, **kwargs)
        
        # Mover módulo de volta para CPU
        self.module.to('cpu')
        
        # Limpar cache CUDA
        torch.cuda.empty_cache()
        
        return result

def apply_cpu_offload(model, offload_layers=None):
    """
    Aplica offloading para CPU em camadas específicas do modelo.
    
    Args:
        model: Modelo a ser modificado
        offload_layers: Lista de índices de camadas para offload (None para todas)
        
    Returns:
        Modelo com offloading habilitado
    """
    if offload_layers is None:
        offload_layers = list(range(len(model.layers)))
    
    # Aplicar wrapper de offload às camadas selecionadas
    for i in offload_layers:
        model.layers[i] = CPUOffloadWrapper(model.layers[i])
    
    return model
```

## Paralelismo e Sharding

Para modelos maiores, podemos dividir o modelo entre múltiplos dispositivos:

### Paralelismo de Dados

```python
def setup_data_parallelism(model, device_ids=None):
    """
    Configura paralelismo de dados para o modelo.
    
    Args:
        model: Modelo a ser paralelizado
        device_ids: Lista de IDs de dispositivos (None para usar todos disponíveis)
        
    Returns:
        Modelo com paralelismo de dados
    """
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    if len(device_ids) <= 1:
        print("Aviso: Paralelismo de dados requer múltiplos dispositivos")
        return model
    
    # Aplicar DataParallel
    model = nn.DataParallel(model, device_ids=device_ids)
    
    return model
```

### Paralelismo de Modelo

```python
class ModelParallelTransformer(nn.Module):
    """
    Implementação de paralelismo de modelo para o Transformer.
    Divide as camadas entre múltiplos dispositivos.
    """
    def __init__(self, model, device_map=None):
        super().__init__()
        self.model = model
        
        # Número de dispositivos disponíveis
        self.num_devices = torch.cuda.device_count()
        
        if self.num_devices <= 1:
            print("Aviso: Paralelismo de modelo requer múltiplos dispositivos")
            self.device_map = {0: list(range(len(model.layers)))}
        else:
            if device_map is None:
                # Distribuir camadas uniformemente entre dispositivos
                layers_per_device = len(model.layers) // self.num_devices
                self.device_map = {}
                
                for i in range(self.num_devices):
                    start_idx = i * layers_per_device
                    end_idx = (i + 1) * layers_per_device if i < self.num_devices - 1 else len(model.layers)
                    self.device_map[i] = list(range(start_idx, end_idx))
            else:
                self.device_map = device_map
        
        # Mover camadas para os dispositivos apropriados
        self.model.embeddings.to(f'cuda:0')
        
        for device_id, layer_indices in self.device_map.items():
            for idx in layer_indices:
                self.model.layers[idx].to(f'cuda:{device_id}')
        
        self.model.ln_f.to(f'cuda:{self.num_devices - 1}')
        self.model.lm_head.to(f'cuda:{self.num_devices - 1}')
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, return_dict=True):
        # Mover entradas para o primeiro dispositivo
        input_ids = input_ids.to('cuda:0')
        if attention_mask is not None:
            attention_mask = attention_mask.to('cuda:0')
        if position_ids is not None:
            position_ids = position_ids.to('cuda:0')
        
        # Obter embeddings (no primeiro dispositivo)
        hidden_states = self.model.embeddings(input_ids, position_ids)
        
        # Criar máscara de atenção se não fornecida
        batch_size, seq_length = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        
        # Converter máscara de atenção
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Inicializar past_key_values se não fornecido
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.model.layers))
        
        all_hidden_states = ()
        all_self_attentions = ()
        next_decoder_cache = ()
        
        # Passar por cada camada, movendo entre dispositivos
        for i, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            # Determinar o dispositivo para esta camada
            device_id = next(iter(device_id for device_id, layer_indices in self.device_map.items() if i in layer_indices))
            
            # Mover estados para o dispositivo correto
            hidden_states = hidden_states.to(f'cuda:{device_id}')
            if extended_attention_mask is not None:
                extended_attention_mask = extended_attention_mask.to(f'cuda:{device_id}')
            
            # Forward pass da camada
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
            
            all_self_attentions = all_self_attentions + (layer_outputs[2],)
        
        # Mover para o último dispositivo para normalização final e projeção
        hidden_states = hidden_states.to(f'cuda:{self.num_devices - 1}')
        
        # Normalização final
        hidden_states = self.model.ln_f(hidden_states)
        
        # Calcular logits
        logits = self.model.lm_head(hidden_states)
        
        if not return_dict:
            return (logits, next_decoder_cache, all_hidden_states, all_self_attentions)
        
        return {
            "logits": logits,
            "past_key_values": next_decoder_cache if use_cache else None,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
        }
```

## Exemplo Prático: Criando um LLM Compacto para GPUs de 4-6GB

Vamos juntar tudo em um exemplo prático para criar e treinar um LLM compacto em uma GPU com memória limitada:

```python
def create_compact_llm(vocab_size=50257, hidden_size=384, num_layers=6, num_heads=6):
    """
    Cria um LLM compacto otimizado para GPUs com memória limitada.
    
    Args:
        vocab_size: Tamanho do vocabulário
        hidden_size: Dimensão dos embeddings e camadas ocultas
        num_layers: Número de camadas do Transformer
        num_heads: Número de cabeças de atenção
        
    Returns:
        Modelo CompactLLM configurado
    """
    # Criar modelo base
    model = CompactLLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,  # Geralmente 4x o hidden_size
        max_position_embeddings=1024,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    
    # Habilitar gradient checkpointing para treinamento eficiente
    model = enable_gradient_checkpointing(model)
    
    return model

def train_compact_llm(
    model,
    train_dataset,
    eval_dataset=None,
    output_dir="./compact_llm",
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_epochs=3,
    warmup_steps=500,
    fp16=True,
    cpu_offload=False,
):
    """
    Treina um LLM compacto com otimizações para GPUs com memória limitada.
    
    Args:
        model: Modelo CompactLLM
        train_dataset: Dataset de treinamento
        eval_dataset: Dataset de avaliação (opcional)
        output_dir: Diretório para salvar o modelo
        batch_size: Tamanho do batch
        gradient_accumulation_steps: Número de passos para acumulação de gradientes
        learning_rate: Taxa de aprendizado
        num_epochs: Número de épocas
        warmup_steps: Número de passos de warmup
        fp16: Se True, usa precisão mista (FP16)
        cpu_offload: Se True, aplica offloading para CPU
        
    Returns:
        Modelo treinado
    """
    from transformers import get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader
    
    # Configurar otimizações de memória
    if cpu_offload:
        model = apply_cpu_offload(model)
    
    # Mover modelo para GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Configurar dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    # Configurar otimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Configurar scheduler
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Configurar scaler para precisão mista
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    # Loop de treinamento
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            # Mover batch para GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Labels são os próprios input_ids para modelos de linguagem causal
            labels = input_ids.clone()
            
            # Forward pass com precisão mista
            if fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs["logits"]
                    
                    # Calcular perda
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass com scaler
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                
                # Calcular perda
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
            
            total_loss += loss.item()
            
            # Atualizar parâmetros a cada gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            # Liberar memória
            torch.cuda.empty_cache()
        
        # Calcular perda média
        avg_loss = total_loss / len(train_dataloader)
        print(f"Época {epoch+1}/{num_epochs}, Perda: {avg_loss:.4f}")
        
        # Avaliação
        if eval_dataset is not None:
            eval_loss = evaluate_model(model, eval_dataset, device, batch_size, fp16)
            print(f"Perda de avaliação: {eval_loss:.4f}")
        
        # Salvar modelo
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(f"{output_dir}/checkpoint-{epoch+1}")
    
    return model

def evaluate_model(model, eval_dataset, device, batch_size=4, fp16=True):
    """
    Avalia o modelo em um dataset de avaliação.
    
    Args:
        model: Modelo CompactLLM
        eval_dataset: Dataset de avaliação
        device: Dispositivo para avaliação
        batch_size: Tamanho do batch
        fp16: Se True, usa precisão mista (FP16)
        
    Returns:
        Perda média de avaliação
    """
    from torch.utils.data import DataLoader
    
    model.eval()
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    total_loss = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Mover batch para GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Labels são os próprios input_ids para modelos de linguagem causal
            labels = input_ids.clone()
            
            # Forward pass com precisão mista
            if fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs["logits"]
                    
                    # Calcular perda
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                
                # Calcular perda
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            
            total_loss += loss.item()
    
    return total_loss / len(eval_dataloader)
```

## Conclusão

Nesta aula, implementamos um LLM compacto otimizado para hardware com limitações de memória. Exploramos:

1. A arquitetura do modelo, baseada no Transformer
2. Técnicas de otimização de memória, como quantização, gradient checkpointing e offloading
3. Estratégias de paralelismo para dividir o modelo entre múltiplos dispositivos
4. Um exemplo prático de como criar e treinar um LLM em GPUs com 4-6GB de memória

Estas técnicas permitem construir e treinar modelos surpreendentemente poderosos mesmo com recursos computacionais limitados. No próximo módulo, exploraremos o processo de treinamento em detalhes, incluindo estratégias para treinar eficientemente em datasets grandes.

## Exercícios Práticos

1. Implemente o modelo CompactLLM e treine-o em um pequeno dataset de texto em português.
2. Experimente com diferentes configurações de tamanho (hidden_size, num_layers, num_heads) e meça o uso de memória.
3. Compare o desempenho do modelo com e sem as técnicas de otimização de memória.
4. Implemente a geração de texto com o modelo treinado e teste diferentes estratégias de sampling.
5. Modifique o código para implementar uma versão do modelo que use LoRA (Low-Rank Adaptation) para fine-tuning eficiente.
