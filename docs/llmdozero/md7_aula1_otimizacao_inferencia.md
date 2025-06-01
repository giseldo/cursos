# Otimização de Inferência

Neste módulo, vamos explorar técnicas para otimizar a inferência de LLMs em hardware com limitações de memória. Aprenderemos a implementar métodos que permitem executar modelos grandes em GPUs com apenas 4-6GB de VRAM ou no Google Colab, mantendo um bom desempenho.

## Técnicas de Quantização para Inferência

A quantização é uma das técnicas mais eficazes para reduzir os requisitos de memória durante a inferência.

### Quantização Pós-Treinamento (PTQ)

A quantização pós-treinamento converte os pesos do modelo de precisão completa (FP32/FP16) para formatos de menor precisão:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def quantize_model_to_int8(model_name, save_dir=None):
    """
    Quantiza um modelo para INT8 usando quantização estática.
    
    Args:
        model_name: Nome ou caminho do modelo
        save_dir: Diretório para salvar o modelo quantizado (opcional)
        
    Returns:
        tuple: (modelo quantizado, tokenizer)
    """
    # Carregar modelo e tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configurar quantização
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Tipos de módulos a serem quantizados
        dtype=torch.qint8    # Tipo de dados para quantização
    )
    
    # Salvar modelo quantizado se especificado
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    
    return model, tokenizer

def quantize_with_bitsandbytes(model_name, bits=8, save_dir=None):
    """
    Quantiza um modelo usando a biblioteca bitsandbytes.
    
    Args:
        model_name: Nome ou caminho do modelo
        bits: Número de bits para quantização (4 ou 8)
        save_dir: Diretório para salvar o modelo quantizado (opcional)
        
    Returns:
        tuple: (modelo quantizado, tokenizer)
    """
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    
    # Configurar quantização
    if bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    elif bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",  # Normalized Float 4
            bnb_4bit_use_double_quant=True
        )
    else:
        raise ValueError(f"Bits não suportados: {bits}. Use 4 ou 8.")
    
    # Carregar modelo quantizado
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"  # Distribui o modelo entre dispositivos disponíveis
    )
    
    # Carregar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Salvar configuração se especificado
    if save_dir:
        import os
        import json
        os.makedirs(save_dir, exist_ok=True)
        
        # Não podemos salvar o modelo quantizado diretamente,
        # mas podemos salvar a configuração para recarregar
        config_path = os.path.join(save_dir, "quantization_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "bits": bits,
                "model_name": model_name
            }, f, indent=2)
        
        # Salvar tokenizer
        tokenizer.save_pretrained(save_dir)
    
    return model, tokenizer
```

### Quantização Consciente de Ativações (AWQ)

A AWQ (Activation-aware Weight Quantization) é uma técnica avançada que considera as ativações durante a quantização:

```python
def explain_awq():
    """
    Explica a técnica AWQ (Activation-aware Weight Quantization).
    """
    explanation = """
    # Activation-aware Weight Quantization (AWQ)
    
    AWQ é uma técnica avançada de quantização que considera as ativações do modelo
    durante o processo de quantização, resultando em melhor precisão com baixa precisão.
    
    ## Princípio Fundamental:
    
    A AWQ se baseia na observação de que nem todos os pesos têm a mesma importância
    para as ativações de saída. Alguns pesos têm impacto muito maior que outros.
    
    ## Como Funciona:
    
    1. **Análise de Sensibilidade**: Identifica quais pesos são mais importantes para
       as ativações de saída em um conjunto de calibração.
       
    2. **Quantização Seletiva**: Preserva a precisão dos pesos mais importantes,
       enquanto quantiza mais agressivamente os menos importantes.
       
    3. **Escala por Canal**: Aplica fatores de escala diferentes para cada canal
       de saída, otimizando a precisão da quantização.
    
    ## Vantagens sobre Quantização Tradicional:
    
    - Melhor preservação da precisão do modelo
    - Menor degradação de desempenho em tarefas complexas
    - Particularmente eficaz para modelos de linguagem grandes
    
    ## Implementação com bibliotecas:
    
    ```python
    # Usando a biblioteca AWQ
    from awq import AutoAWQForCausalLM
    
    # Carregar modelo em precisão completa
    model_path = "meta-llama/Llama-2-7b-hf"
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    
    # Configurar quantização
    quant_config = {
        "zero_point": True,      # Usar zero point para melhor precisão
        "q_group_size": 128,     # Tamanho do grupo para quantização
        "w_bit": 4,              # Bits para quantização (4 ou 8)
        "version": "GEMM"        # Versão do kernel de inferência
    }
    
    # Preparar dataset de calibração
    calibration_dataset = [
        "Este é um exemplo de texto para calibração.",
        "A quantização AWQ considera as ativações do modelo."
    ]
    
    # Quantizar modelo
    model.quantize(
        tokenizer,
        calibration_dataset,
        quant_config=quant_config
    )
    
    # Salvar modelo quantizado
    model.save_quantized("./model_awq_4bit")
    ```
    
    ## Resultados Típicos:
    
    | Modelo    | Bits | Tamanho Original | Tamanho Quantizado | Perda Relativa |
    |-----------|------|------------------|-------------------|----------------|
    | LLaMA-7B  | 4    | 13 GB            | 3.5 GB            | ~0.5% em perplexidade |
    | LLaMA-13B | 4    | 25 GB            | 6.5 GB            | ~0.8% em perplexidade |
    | LLaMA-70B | 4    | 140 GB           | 35 GB             | ~1.2% em perplexidade |
    """
    
    return explanation
```

### GPTQ: Quantização Otimizada para Transformers

GPTQ é uma técnica de quantização específica para modelos Transformer:

```python
def explain_gptq():
    """
    Explica a técnica GPTQ (Generative Pretrained Transformer Quantization).
    """
    explanation = """
    # GPTQ (Generative Pretrained Transformer Quantization)
    
    GPTQ é uma técnica de quantização otimizada especificamente para modelos Transformer,
    que permite quantização de alta precisão com apenas 3-4 bits por peso.
    
    ## Princípio Fundamental:
    
    GPTQ usa aproximação de mínimos quadrados com uma heurística de reordenação
    para minimizar o erro introduzido pela quantização.
    
    ## Como Funciona:
    
    1. **Reordenação Inteligente**: Reordena os pesos para minimizar o erro de quantização
       usando uma heurística baseada na matriz de Hessian.
       
    2. **Quantização Sequencial**: Quantiza os pesos um por um, atualizando os pesos
       restantes para compensar o erro introduzido.
       
    3. **Otimização por Camada**: Aplica a quantização separadamente para cada camada
       do modelo, preservando a estrutura do Transformer.
    
    ## Vantagens sobre Outras Técnicas:
    
    - Melhor precisão em bits ultra-baixos (3-4 bits)
    - Processo de quantização mais rápido que métodos iterativos
    - Especialmente eficaz para modelos de linguagem grandes
    
    ## Implementação com bibliotecas:
    
    ```python
    # Usando a biblioteca AutoGPTQ
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    
    # Configurar quantização
    quantize_config = BaseQuantizeConfig(
        bits=4,                # Bits para quantização
        group_size=128,        # Tamanho do grupo
        desc_act=False,        # Usar descritor de ativação
    )
    
    # Carregar modelo para quantização
    model = AutoGPTQForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantize_config
    )
    
    # Preparar dataset de exemplos
    examples = [
        "GPTQ é uma técnica de quantização para modelos de linguagem.",
        "A quantização reduz o tamanho do modelo preservando seu desempenho."
    ]
    
    # Quantizar modelo
    model.quantize(
        examples,
        use_triton=False
    )
    
    # Salvar modelo quantizado
    model.save_quantized("./model_gptq_4bit")
    
    # Carregar modelo quantizado para inferência
    model = AutoGPTQForCausalLM.from_quantized(
        "./model_gptq_4bit",
        device="cuda:0"
    )
    ```
    
    ## Resultados Típicos:
    
    | Modelo    | Bits | Tamanho Original | Tamanho Quantizado | Perda Relativa |
    |-----------|------|------------------|-------------------|----------------|
    | LLaMA-7B  | 4    | 13 GB            | 3.5 GB            | ~0.3% em perplexidade |
    | LLaMA-13B | 4    | 25 GB            | 6.5 GB            | ~0.5% em perplexidade |
    | LLaMA-70B | 4    | 140 GB           | 35 GB             | ~0.8% em perplexidade |
    | LLaMA-7B  | 3    | 13 GB            | 2.6 GB            | ~1.0% em perplexidade |
    """
    
    return explanation
```

## Técnicas de Offloading

O offloading move partes do modelo entre dispositivos para economizar memória.

### CPU Offloading

```python
class CPUOffloadModule(torch.nn.Module):
    """
    Wrapper para offloading de módulos para CPU.
    """
    def __init__(self, module):
        """
        Inicializa o wrapper de offloading.
        
        Args:
            module: Módulo a ser offloaded
        """
        super().__init__()
        self.module = module
        self.cpu_device = torch.device("cpu")
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, *args, **kwargs):
        """
        Forward pass com offloading.
        
        Args:
            *args, **kwargs: Argumentos para o módulo
            
        Returns:
            Resultado do módulo
        """
        # Mover módulo para GPU
        self.module.to(self.gpu_device)
        
        # Mover argumentos para GPU
        args_gpu = [a.to(self.gpu_device) if isinstance(a, torch.Tensor) else a for a in args]
        kwargs_gpu = {k: v.to(self.gpu_device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        
        # Forward pass
        outputs = self.module(*args_gpu, **kwargs_gpu)
        
        # Mover módulo de volta para CPU
        self.module.to(self.cpu_device)
        
        # Limpar cache CUDA
        torch.cuda.empty_cache()
        
        # Mover saídas para CPU se necessário
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.to(self.gpu_device)
        elif isinstance(outputs, tuple):
            outputs = tuple(o.to(self.gpu_device) if isinstance(o, torch.Tensor) else o for o in outputs)
        elif isinstance(outputs, list):
            outputs = [o.to(self.gpu_device) if isinstance(o, torch.Tensor) else o for o in outputs]
        elif isinstance(outputs, dict):
            outputs = {k: v.to(self.gpu_device) if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}
        
        return outputs

def apply_cpu_offloading(model, layer_indices=None):
    """
    Aplica CPU offloading a camadas específicas de um modelo.
    
    Args:
        model: Modelo a ser modificado
        layer_indices: Índices das camadas para aplicar offloading (None para automático)
        
    Returns:
        nn.Module: Modelo com offloading aplicado
    """
    # Se layer_indices não for especificado, determinar automaticamente
    if layer_indices is None:
        # Estimar memória disponível
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_memory_gb = free_memory / (1024**3)
            
            # Estimar número de camadas que cabem na memória
            # Assumindo que cada camada usa aproximadamente a mesma quantidade de memória
            num_layers = len(model.layers) if hasattr(model, "layers") else 0
            if num_layers > 0:
                # Manter metade das camadas na GPU, offload o resto
                layers_to_keep = max(1, int(num_layers * min(0.5, free_memory_gb / 10)))
                layer_indices = list(range(layers_to_keep, num_layers))
        else:
            # Se não houver GPU, não aplicar offloading
            return model
    
    # Aplicar offloading às camadas especificadas
    if hasattr(model, "layers"):
        for i in layer_indices:
            if i < len(model.layers):
                model.layers[i] = CPUOffloadModule(model.layers[i])
    
    return model

def disk_offloading_example():
    """
    Exemplo de código para offloading para disco.
    """
    example_code = """
    import torch
    import os
    
    class DiskOffloadModule(torch.nn.Module):
        """
        Wrapper para offloading de módulos para disco.
        """
        def __init__(self, module, temp_dir="./.offload_temp"):
            super().__init__()
            self.module = module
            self.temp_dir = temp_dir
            self.module_path = os.path.join(temp_dir, "module.pt")
            
            # Criar diretório temporário
            os.makedirs(temp_dir, exist_ok=True)
            
            # Salvar módulo em disco
            self._save_to_disk()
        
        def _save_to_disk(self):
            """Salva o módulo em disco e libera memória."""
            torch.save(self.module.state_dict(), self.module_path)
            # Limpar parâmetros para liberar memória
            for param in self.module.parameters():
                param.data = torch.zeros(1)  # Placeholder mínimo
        
        def _load_from_disk(self):
            """Carrega o módulo do disco."""
            self.module.load_state_dict(torch.load(self.module_path))
        
        def forward(self, *args, **kwargs):
            # Carregar módulo do disco
            self._load_from_disk()
            
            # Forward pass
            outputs = self.module(*args, **kwargs)
            
            # Salvar módulo de volta para disco e liberar memória
            self._save_to_disk()
            
            return outputs
    """
    
    return example_code
```

### Inferência com Atenção em Janela Deslizante

A atenção em janela deslizante permite processar sequências longas com memória limitada:

```python
def sliding_window_attention(model, input_ids, attention_mask=None, window_size=1024, stride=512, device=None):
    """
    Realiza inferência com atenção em janela deslizante para sequências longas.
    
    Args:
        model: Modelo para inferência
        input_ids: Tensor de índices de tokens
        attention_mask: Máscara de atenção opcional
        window_size: Tamanho da janela de atenção
        stride: Passo entre janelas consecutivas
        device: Dispositivo para inferência
        
    Returns:
        torch.Tensor: Estados ocultos para toda a sequência
    """
    # Configurar device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mover modelo para device
    model.to(device)
    model.eval()
    
    # Obter dimensões
    batch_size, seq_length = input_ids.size()
    
    # Verificar se a sequência é menor que a janela
    if seq_length <= window_size:
        # Processar toda a sequência de uma vez
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device) if attention_mask is not None else None,
                output_hidden_states=True
            )
        
        # Retornar estados ocultos da última camada
        return outputs.hidden_states[-1]
    
    # Inicializar tensor para armazenar estados ocultos
    hidden_size = model.config.hidden_size
    all_hidden_states = torch.zeros((batch_size, seq_length, hidden_size), device=device)
    
    # Processar a sequência em janelas
    for start_idx in range(0, seq_length, stride):
        # Definir índices da janela
        end_idx = min(start_idx + window_size, seq_length)
        
        # Extrair janela
        window_input_ids = input_ids[:, start_idx:end_idx]
        window_attention_mask = attention_mask[:, start_idx:end_idx] if attention_mask is not None else None
        
        # Processar janela
        with torch.no_grad():
            outputs = model(
                input_ids=window_input_ids.to(device),
                attention_mask=window_attention_mask.to(device) if window_attention_mask is not None else None,
                output_hidden_states=True
            )
        
        # Extrair estados ocultos da última camada
        window_hidden_states = outputs.hidden_states[-1]
        
        # Determinar região válida (evitar sobreposição)
        if start_idx == 0:
            # Primeira janela: usar tudo até stride
            valid_end_idx = min(stride, end_idx)
            all_hidden_states[:, :valid_end_idx] = window_hidden_states[:, :valid_end_idx]
        elif end_idx == seq_length:
            # Última janela: usar a partir do último ponto válido
            last_valid_idx = start_idx
            all_hidden_states[:, last_valid_idx:] = window_hidden_states[:, -(seq_length - last_valid_idx):]
        else:
            # Janelas intermediárias: usar região central
            valid_start_idx = start_idx
            valid_end_idx = min(start_idx + stride, end_idx)
            window_start_idx = valid_start_idx - start_idx
            window_end_idx = valid_end_idx - start_idx
            all_hidden_states[:, valid_start_idx:valid_end_idx] = window_hidden_states[:, window_start_idx:window_end_idx]
    
    return all_hidden_states
```

## Geração Eficiente de Texto

A geração de texto pode ser otimizada para hardware limitado.

### Geração com KV Cache Otimizado

```python
def optimized_generate(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.0,
    do_sample=True,
    num_return_sequences=1,
    device=None,
    use_kv_cache=True
):
    """
    Gera texto de forma otimizada para hardware limitado.
    
    Args:
        model: Modelo para geração
        tokenizer: Tokenizer do modelo
        prompt: Texto de prompt
        max_length: Comprimento máximo da sequência gerada
        temperature: Temperatura para amostragem
        top_p: Probabilidade cumulativa para nucleus sampling
        top_k: Número de tokens mais prováveis a considerar
        repetition_penalty: Penalidade para repetição de tokens
        do_sample: Se True, amostra da distribuição, caso contrário usa greedy decoding
        num_return_sequences: Número de sequências a retornar
        device: Dispositivo para geração
        use_kv_cache: Se True, usa cache de chaves e valores para geração mais rápida
        
    Returns:
        list: Lista de textos gerados
    """
    # Configurar device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mover modelo para device
    model.to(device)
    model.eval()
    
    # Tokenizar prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Inicializar cache KV
    past_key_values = None
    
    # Gerar sequências
    generated_sequences = []
    
    for _ in range(num_return_sequences):
        # Copiar input_ids para esta sequência
        curr_input_ids = input_ids.clone()
        
        # Inicializar cache KV para esta sequência
        curr_past_key_values = past_key_values
        
        # Gerar tokens autoregressivamente
        for _ in range(max_length):
            # Forward pass
            with torch.no_grad():
                if use_kv_cache:
                    # Usar apenas o último token se temos cache
                    if curr_past_key_values is not None:
                        token_input_ids = curr_input_ids[:, -1].unsqueeze(-1)
                    else:
                        token_input_ids = curr_input_ids
                    
                    outputs = model(
                        input_ids=token_input_ids,
                        past_key_values=curr_past_key_values,
                        use_cache=True
                    )
                    
                    # Atualizar cache KV
                    curr_past_key_values = outputs.past_key_values
                else:
                    # Processar toda a sequência
                    outputs = model(input_ids=curr_input_ids)
                
                # Obter logits do próximo token
                next_token_logits = outputs.logits[:, -1, :]
            
            # Aplicar temperatura
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Aplicar repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(curr_input_ids[0].tolist()):
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty
            
            # Filtrar com top-k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("inf")
            
            # Filtrar com top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens com probabilidade cumulativa acima do threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift para manter também o primeiro token acima do threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[0, indices_to_remove] = -float("inf")
            
            # Amostrar próximo token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Adicionar token à sequência
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
            
            # Verificar se atingimos token de fim
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        # Decodificar sequência
        generated_text = tokenizer.decode(curr_input_ids[0], skip_special_tokens=True)
        generated_sequences.append(generated_text)
    
    return generated_sequences
```

### Geração em Lote com Memória Limitada

```python
def batch_generate_with_limited_memory(
    model,
    tokenizer,
    prompts,
    max_batch_size=4,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    device=None
):
    """
    Gera texto para múltiplos prompts em lotes, otimizado para memória limitada.
    
    Args:
        model: Modelo para geração
        tokenizer: Tokenizer do modelo
        prompts: Lista de prompts
        max_batch_size: Tamanho máximo do lote
        max_length: Comprimento máximo da sequência gerada
        temperature: Temperatura para amostragem
        top_p: Probabilidade cumulativa para nucleus sampling
        device: Dispositivo para geração
        
    Returns:
        list: Lista de textos gerados
    """
    # Configurar device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mover modelo para device
    model.to(device)
    model.eval()
    
    # Inicializar lista para resultados
    results = []
    
    # Processar prompts em lotes
    for i in range(0, len(prompts), max_batch_size):
        # Extrair lote atual
        batch_prompts = prompts[i:i+max_batch_size]
        
        # Tokenizar prompts
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
        
        # Gerar texto
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_inputs.input_ids,
                attention_mask=batch_inputs.attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decodificar saídas
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)
        
        # Limpar cache CUDA
        torch.cuda.empty_cache()
    
    return results
```

## Otimização com ONNX Runtime

ONNX Runtime pode acelerar significativamente a inferência:

```python
def convert_to_onnx(model, tokenizer, onnx_path, device=None):
    """
    Converte um modelo PyTorch para formato ONNX.
    
    Args:
        model: Modelo PyTorch
        tokenizer: Tokenizer do modelo
        onnx_path: Caminho para salvar o modelo ONNX
        device: Dispositivo para conversão
        
    Returns:
        str: Caminho do modelo ONNX
    """
    import os
    import torch.onnx
    
    # Configurar device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mover modelo para device e modo de avaliação
    model.to(device)
    model.eval()
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # Criar entrada de exemplo
    dummy_input = tokenizer("Exemplo de texto para conversão ONNX", return_tensors="pt").to(device)
    
    # Exportar para ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,                                      # modelo
            (dummy_input.input_ids, dummy_input.attention_mask),  # entradas
            onnx_path,                                  # caminho de saída
            export_params=True,                         # armazenar os parâmetros treinados
            opset_version=13,                           # versão do operador ONNX
            do_constant_folding=True,                   # otimização
            input_names=["input_ids", "attention_mask"],  # nomes das entradas
            output_names=["logits"],                    # nomes das saídas
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            }
        )
    
    print(f"Modelo convertido para ONNX e salvo em {onnx_path}")
    return onnx_path

def optimize_onnx_model(onnx_path, optimized_path=None):
    """
    Otimiza um modelo ONNX para inferência.
    
    Args:
        onnx_path: Caminho do modelo ONNX
        optimized_path: Caminho para salvar o modelo otimizado (opcional)
        
    Returns:
        str: Caminho do modelo otimizado
    """
    import onnx
    from onnxruntime.transformers import optimizer
    
    # Definir caminho de saída se não especificado
    if optimized_path is None:
        optimized_path = onnx_path.replace(".onnx", "_optimized.onnx")
    
    # Carregar modelo ONNX
    model = onnx.load(onnx_path)
    
    # Otimizar modelo
    optimized_model = optimizer.optimize_model(
        onnx_path,
        model_type="bert",  # ou "gpt2" para modelos causais
        num_heads=12,       # número de cabeças de atenção
        hidden_size=768     # tamanho oculto do modelo
    )
    
    # Salvar modelo otimizado
    optimized_model.save_model_to_file(optimized_path)
    
    print(f"Modelo ONNX otimizado e salvo em {optimized_path}")
    return optimized_path

def inference_with_onnx(onnx_path, tokenizer, text, device="cpu"):
    """
    Realiza inferência com um modelo ONNX.
    
    Args:
        onnx_path: Caminho do modelo ONNX
        tokenizer: Tokenizer do modelo
        text: Texto para inferência
        device: Dispositivo para inferência ("cpu" ou "cuda")
        
    Returns:
        np.ndarray: Logits de saída
    """
    import numpy as np
    import onnxruntime as ort
    
    # Configurar sessão ONNX Runtime
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers = ["CUDAExecutionProvider"] + providers
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Tokenizar entrada
    inputs = tokenizer(text, return_tensors="pt")
    
    # Converter para numpy
    onnx_inputs = {
        "input_ids": inputs.input_ids.numpy(),
        "attention_mask": inputs.attention_mask.numpy()
    }
    
    # Realizar inferência
    outputs = session.run(None, onnx_inputs)
    
    # Retornar logits
    return outputs[0]  # Assumindo que logits é a primeira saída
```

## Inferência com TensorRT

TensorRT pode oferecer aceleração significativa em GPUs NVIDIA:

```python
def explain_tensorrt():
    """
    Explica como usar TensorRT para acelerar inferência.
    """
    explanation = """
    # Aceleração de Inferência com TensorRT
    
    TensorRT é uma plataforma de alto desempenho da NVIDIA para inferência de deep learning
    que pode acelerar significativamente a execução de modelos em GPUs NVIDIA.
    
    ## Benefícios do TensorRT:
    
    1. **Otimização de Kernel**: Seleciona automaticamente os kernels mais eficientes
    2. **Fusão de Camadas**: Combina operações para reduzir transferências de memória
    3. **Precisão Mista**: Suporta FP32, FP16 e INT8 para balancear precisão e velocidade
    4. **Otimização de Memória**: Gerencia eficientemente o uso de memória
    
    ## Processo de Implementação:
    
    ```python
    # Instalar bibliotecas necessárias
    !pip install torch tensorrt
    
    import torch
    import tensorrt as trt
    from torch2trt import torch2trt
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Carregar modelo e tokenizer
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Mover para GPU
    model = model.cuda().eval()
    
    # Criar entrada de exemplo
    dummy_input = tokenizer("Exemplo para conversão TensorRT", return_tensors="pt").to("cuda")
    
    # Converter para TensorRT
    trt_model = torch2trt(
        model,
        [dummy_input.input_ids, dummy_input.attention_mask],
        fp16_mode=True,
        max_batch_size=8,
        max_workspace_size=1 << 30  # 1GB
    )
    
    # Salvar modelo TensorRT
    torch.save(trt_model.state_dict(), "model_trt.pth")
    
    # Carregar modelo TensorRT
    from torch2trt import TRTModule
    trt_model = TRTModule()
    trt_model.load_state_dict(torch.load("model_trt.pth"))
    
    # Inferência com TensorRT
    with torch.no_grad():
        outputs = trt_model(dummy_input.input_ids, dummy_input.attention_mask)
    ```
    
    ## Considerações para GPUs com Memória Limitada:
    
    1. **Quantização INT8**: Reduz significativamente o uso de memória
       ```python
       trt_model = torch2trt(model, [dummy_input], int8_mode=True, int8_calib_dataset=calibration_dataset)
       ```
    
    2. **Segmentação de Modelo**: Divide o modelo em partes menores
       ```python
       # Converter apenas algumas camadas para TensorRT
       trt_layers = []
       for i in range(len(model.layers)):
           trt_layer = torch2trt(model.layers[i], [dummy_layer_input], fp16_mode=True)
           trt_layers.append(trt_layer)
       ```
    
    3. **Otimização de Workspace**: Limitar o espaço de trabalho para GPUs menores
       ```python
       trt_model = torch2trt(model, [dummy_input], fp16_mode=True, max_workspace_size=1 << 28)  # 256MB
       ```
    
    ## Ganhos de Desempenho Típicos:
    
    | Modelo    | Precisão | Speedup vs. PyTorch | Redução de Memória |
    |-----------|----------|---------------------|-------------------|
    | GPT-2 Small | FP16   | 2-3x                | ~30%              |
    | GPT-2 Small | INT8   | 3-4x                | ~60%              |
    | BERT Base   | FP16   | 2-4x                | ~30%              |
    | BERT Base   | INT8   | 4-6x                | ~60%              |
    """
    
    return explanation
```

## Otimização para Google Colab

Técnicas específicas para otimizar a execução no Google Colab:

```python
def optimize_for_colab(model_name, use_8bit=True, use_4bit=False, cpu_offload=False):
    """
    Otimiza um modelo para execução no Google Colab.
    
    Args:
        model_name: Nome ou caminho do modelo
        use_8bit: Se True, usa quantização de 8 bits
        use_4bit: Se True, usa quantização de 4 bits (tem precedência sobre 8 bits)
        cpu_offload: Se True, aplica offloading para CPU
        
    Returns:
        tuple: (modelo otimizado, tokenizer)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Verificar memória disponível
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / (1024**3)
        print(f"Memória GPU disponível: {gpu_memory_gb:.2f} GB")
    else:
        print("GPU não disponível, usando CPU")
        gpu_memory_gb = 0
    
    # Configurar quantização com base na memória disponível
    if gpu_memory_gb < 4 or not torch.cuda.is_available():
        # Memória muito limitada, forçar offloading
        cpu_offload = True
        if use_4bit:
            print("Usando quantização de 4 bits com offloading para CPU")
        elif use_8bit:
            print("Usando quantização de 8 bits com offloading para CPU")
        else:
            print("Usando offloading para CPU sem quantização")
    elif gpu_memory_gb < 8:
        # Memória limitada, recomendar quantização mais agressiva
        if not use_4bit and not use_8bit:
            print("Aviso: Memória GPU limitada, recomenda-se usar quantização")
    
    # Carregar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Garantir que o tokenizer tenha token de padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Carregar modelo com otimizações
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            
            # Configuração para quantização de 4 bits
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            # Carregar modelo quantizado
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if not cpu_offload else {"": "cpu"}
            )
            
            print("Modelo carregado com quantização de 4 bits")
        except ImportError:
            print("Biblioteca bitsandbytes não disponível, usando quantização de 8 bits")
            use_8bit = True
            use_4bit = False
    
    if use_8bit and not use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            
            # Configuração para quantização de 8 bits
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            # Carregar modelo quantizado
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if not cpu_offload else {"": "cpu"}
            )
            
            print("Modelo carregado com quantização de 8 bits")
        except ImportError:
            print("Biblioteca bitsandbytes não disponível, usando modelo sem quantização")
            model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if not use_4bit and not use_8bit:
        # Carregar modelo sem quantização
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Modelo carregado sem quantização")
    
    # Aplicar CPU offloading se solicitado
    if cpu_offload and not (use_4bit or use_8bit):  # Quantização já lida com offloading
        model = apply_cpu_offloading(model)
        print("Offloading para CPU aplicado")
    
    return model, tokenizer

def colab_inference_example():
    """
    Exemplo completo de inferência otimizada para Google Colab.
    """
    example_code = """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    # Função para verificar memória disponível
    def check_available_memory():
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            free_memory = gpu_memory - allocated_memory
            print(f"Memória GPU total: {gpu_memory / 1e9:.2f} GB")
            print(f"Memória GPU alocada: {allocated_memory / 1e9:.2f} GB")
            print(f"Memória GPU livre: {free_memory / 1e9:.2f} GB")
            return free_memory / 1e9  # GB
        else:
            print("GPU não disponível")
            return 0
    
    # Verificar memória antes de carregar o modelo
    free_memory_gb = check_available_memory()
    
    # Escolher modelo com base na memória disponível
    if free_memory_gb >= 12:
        model_name = "meta-llama/Llama-2-7b-hf"  # Modelo maior
        use_4bit = False
        use_8bit = True
    elif free_memory_gb >= 6:
        model_name = "meta-llama/Llama-2-7b-hf"  # Mesmo modelo, mais comprimido
        use_4bit = True
        use_8bit = False
    else:
        model_name = "distilgpt2"  # Modelo menor
        use_4bit = False
        use_8bit = False
    
    print(f"Usando modelo: {model_name}")
    
    # Configurar quantização se necessário
    if use_4bit:
        print("Usando quantização de 4 bits")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
    elif use_8bit:
        print("Usando quantização de 8 bits")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        print("Carregando modelo sem quantização")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Mover para GPU se disponível
        if torch.cuda.is_available():
            model = model.to("cuda")
    
    # Carregar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Verificar memória após carregar o modelo
    check_available_memory()
    
    # Função para geração de texto otimizada
    def generate_text(prompt, max_length=100, temperature=0.7, top_p=0.9):
        # Tokenizar prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Mover para o mesmo dispositivo do modelo
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # Gerar texto
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decodificar saída
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Limpar cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return generated_text
    
    # Exemplo de uso
    prompt = "Explique como criar um modelo de linguagem em Python:"
    generated_text = generate_text(prompt)
    print(generated_text)
    """
    
    return example_code
```

## Conclusão

Neste módulo, exploramos técnicas para otimizar a inferência de LLMs em hardware com limitações de memória. Aprendemos a:

1. Aplicar diferentes técnicas de quantização (PTQ, AWQ, GPTQ) para reduzir o tamanho do modelo
2. Implementar offloading para CPU e disco para lidar com modelos maiores que a memória disponível
3. Otimizar a geração de texto com cache KV e processamento em lote
4. Acelerar a inferência com ONNX Runtime e TensorRT
5. Adaptar modelos para execução eficiente no Google Colab

Estas técnicas permitem executar modelos surpreendentemente grandes em hardware com limitações, abrindo possibilidades para experimentação e uso prático de LLMs em uma variedade de dispositivos.

## Exercícios Práticos

1. Compare o desempenho e uso de memória de um modelo (como GPT-2 medium) com diferentes níveis de quantização (FP16, INT8, INT4).
2. Implemente a geração de texto com atenção em janela deslizante e compare com a geração padrão para sequências longas.
3. Converta um modelo pequeno para ONNX e compare o tempo de inferência com PyTorch.
4. Implemente CPU offloading para um modelo e meça o impacto no tempo de inferência e uso de memória.
5. Crie um notebook para Google Colab que carregue e execute um modelo de 7B parâmetros usando as técnicas de otimização discutidas.
