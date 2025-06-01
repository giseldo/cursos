# Requisitos de Hardware e Otimizações para LLMs

Trabalhar com Modelos de Linguagem de Grande Escala (LLMs) tradicionalmente exige recursos computacionais substanciais, especialmente em termos de memória de GPU. No entanto, com as técnicas certas de otimização, é possível construir, treinar e utilizar LLMs mesmo em hardware com limitações de memória, como GPUs de 4-6GB ou no ambiente do Google Colab. Nesta aula, exploraremos os requisitos de hardware para LLMs e as estratégias de otimização que tornam possível trabalhar com esses modelos em configurações mais modestas.

## Entendendo as Limitações de Memória

Antes de discutirmos as otimizações, é importante compreender por que os LLMs consomem tanta memória e quais são os principais gargalos.

### Principais Consumidores de Memória em LLMs

1. **Parâmetros do Modelo**: Os pesos e biases que definem o modelo. Um LLM típico pode ter de milhões a bilhões de parâmetros.

2. **Estados de Ativação**: Os valores intermediários calculados durante o forward pass, que precisam ser armazenados para o cálculo do gradiente durante o backward pass.

3. **Otimizadores**: Algoritmos como Adam mantêm estatísticas adicionais para cada parâmetro (momentos de primeira e segunda ordem), efetivamente multiplicando o requisito de memória.

4. **Batch Size**: O número de exemplos processados simultaneamente, que multiplica linearmente o consumo de memória das ativações.

5. **Precisão dos Parâmetros**: O formato numérico usado para representar os parâmetros (FP32, FP16, INT8, etc.).

### Cálculo Aproximado de Requisitos de Memória

Para um modelo Transformer com \(n\) parâmetros, usando precisão de ponto flutuante de 32 bits (FP32) e o otimizador Adam, podemos estimar o consumo de memória durante o treinamento:

- Parâmetros: \(n \times 4\) bytes
- Estados do otimizador: \(n \times 4 \times 2\) bytes (para Adam)
- Gradientes: \(n \times 4\) bytes
- Ativações: Depende da arquitetura, tamanho de entrada e batch size

Por exemplo, um modelo com 100 milhões de parâmetros em FP32 com Adam requer aproximadamente:
- 100M × 4 bytes = 400MB para os parâmetros
- 100M × 4 × 2 bytes = 800MB para os estados do otimizador
- 100M × 4 bytes = 400MB para os gradientes
Total: 1.6GB apenas para parâmetros, otimizador e gradientes, sem contar as ativações!

## Técnicas de Otimização para Hardware Limitado

Felizmente, existem várias técnicas que podemos empregar para reduzir significativamente os requisitos de memória:

### 1. Redução de Precisão Numérica

Uma das técnicas mais eficazes é usar formatos numéricos de menor precisão:

- **FP16 (Half Precision)**: Usa 16 bits em vez de 32 bits por parâmetro, reduzindo o consumo de memória pela metade.
- **BF16 (Brain Floating Point)**: Similar ao FP16, mas com melhor alcance numérico.
- **INT8 Quantização**: Usa inteiros de 8 bits, reduzindo o consumo de memória para 1/4 do FP32.

```python
# Exemplo de conversão para FP16 em PyTorch
model = model.half()  # Converte o modelo para FP16
```

### 2. Gradient Checkpointing

Esta técnica sacrifica velocidade de computação para economizar memória. Em vez de armazenar todas as ativações intermediárias, armazenamos apenas algumas e recalculamos as outras durante o backward pass quando necessário.

```python
# Exemplo de gradient checkpointing em PyTorch
from torch.utils.checkpoint import checkpoint

# Definindo um módulo que usa checkpointing
class CheckpointedModule(nn.Module):
    def __init__(self, submodule):
        super().__init__()
        self.submodule = submodule
        
    def forward(self, x):
        return checkpoint(self.submodule, x)
```

O gradient checkpointing pode reduzir o consumo de memória das ativações em até 80%, com um custo de aproximadamente 30% em tempo de computação.

### 3. Offloading para CPU

Podemos mover temporariamente partes do modelo ou estados do otimizador para a RAM da CPU quando não estão em uso ativo, liberando memória de GPU.

```python
# Exemplo simplificado de offloading manual
# Mover parâmetros para CPU
cpu_params = [p.cpu() for p in model.parameters()]
# Liberar memória na GPU
for p in model.parameters():
    p.data = torch.empty(0, device='cuda')
# ... fazer algum processamento que requer memória ...
# Restaurar parâmetros para GPU
for i, p in enumerate(model.parameters()):
    p.data = cpu_params[i].cuda()
```

Bibliotecas como `accelerate` da Hugging Face automatizam esse processo:

```python
from accelerate import Accelerator

accelerator = Accelerator(cpu_offload_model=True)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

### 4. Acumulação de Gradientes

Em vez de aumentar o batch size diretamente (o que aumenta o consumo de memória), podemos acumular gradientes ao longo de vários mini-batches menores antes de atualizar os parâmetros.

```python
# Exemplo de acumulação de gradientes
accumulation_steps = 8  # Número de passos para acumular
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = loss_function(outputs)
    # Normalizar a perda pelo número de passos de acumulação
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Esta técnica permite simular batches maiores sem o correspondente aumento no consumo de memória.

### 5. Arquiteturas Eficientes

Algumas variantes de arquiteturas Transformer foram projetadas especificamente para eficiência:

- **Reformer**: Usa hashing sensível à localidade para reduzir a complexidade da atenção.
- **Linformer**: Reduz a complexidade da atenção de O(n²) para O(n) usando projeções de baixo rank.
- **Performer**: Usa aproximações de kernel para o mecanismo de atenção.

```python
# Exemplo de uso do Reformer (usando a biblioteca Hugging Face)
from transformers import ReformerConfig, ReformerModel

config = ReformerConfig(
    attention_head_size=64,
    axial_pos_embds=True,
    axial_pos_shape=[64, 64],
    axial_pos_embds_dim=[64, 192],
    chunk_size_feed_forward=0,
    eos_token_id=2,
    feed_forward_size=512,
    hash_seed=None,
    hidden_size=256,
    num_attention_heads=2,
    num_buckets=None,
    num_hashes=1,
    pad_token_id=0,
    vocab_size=320
)

model = ReformerModel(config)
```

### 6. Sharding e Paralelismo

Para modelos maiores, podemos dividir o modelo entre múltiplos dispositivos:

- **Paralelismo de Dados**: Cada dispositivo tem uma cópia completa do modelo, mas processa diferentes batches de dados.
- **Paralelismo de Modelo**: O modelo é dividido entre dispositivos, com cada um responsável por diferentes camadas.
- **Paralelismo de Tensor**: Os parâmetros individuais são divididos entre dispositivos.

```python
# Exemplo de paralelismo de dados com PyTorch
model = torch.nn.DataParallel(model)
```

### 7. Técnicas Específicas para Treinamento vs. Inferência

Durante o treinamento, precisamos armazenar ativações para o backward pass, mas na inferência podemos descartar ativações imediatamente após o uso.

Para inferência, técnicas adicionais incluem:
- **Pruning**: Remover conexões menos importantes do modelo.
- **Quantização Pós-Treinamento**: Quantizar o modelo após o treinamento para inferência mais eficiente.
- **Distilação de Conhecimento**: Treinar um modelo menor para imitar um modelo maior.

## Configuração do Ambiente no Google Colab vs. Máquina Local

Vamos comparar as configurações para trabalhar com LLMs em diferentes ambientes:

### Google Colab

**Vantagens**:
- Acesso gratuito a GPUs (T4, P100, ocasionalmente V100)
- Ambiente pré-configurado com bibliotecas essenciais
- Integração com Google Drive para armazenamento persistente
- Sem necessidade de configuração de hardware

**Limitações**:
- Tempo de execução limitado (12h para Colab Pro)
- Desconexões ocasionais
- Alocação de GPU não garantida
- Memória de GPU limitada (geralmente 12-16GB)

**Configuração Recomendada**:
```python
# Verificar GPU disponível
!nvidia-smi

# Instalar bibliotecas adicionais
!pip install transformers accelerate bitsandbytes

# Montar Google Drive para armazenamento persistente
from google.colab import drive
drive.mount('/content/drive')

# Configurar para economia de memória
import torch
from accelerate import Accelerator

# Usar mixed precision
accelerator = Accelerator(mixed_precision='fp16')

# Verificar memória disponível
!nvidia-smi
```

### Máquina Local com GPU de 4-6GB

**Vantagens**:
- Controle total sobre o ambiente
- Sem limitações de tempo de execução
- Acesso consistente aos recursos
- Possibilidade de otimizações específicas para o hardware

**Limitações**:
- Memória de GPU muito limitada
- Pode exigir configuração mais complexa
- Treinamento mais lento comparado a GPUs profissionais

**Configuração Recomendada**:
```python
# Configuração para GPU com memória limitada
import torch
import os

# Definir GPU a ser usada (se houver múltiplas)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Limitar cache CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Configurar para economia de memória
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Usar mixed precision
scaler = torch.cuda.amp.GradScaler()

# Função para monitorar uso de memória
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Estratégias Práticas para Trabalhar com LLMs em Hardware Limitado

Combinando as técnicas discutidas, aqui estão algumas estratégias práticas para diferentes cenários:

### Para Treinamento de Modelos Pequenos a Médios (até 100M parâmetros)

1. Use precisão mista (FP16/BF16)
2. Implemente gradient checkpointing
3. Use batch sizes pequenos (1-4) com acumulação de gradientes
4. Considere arquiteturas eficientes como Reformer ou Linformer
5. Monitore cuidadosamente o uso de memória

### Para Fine-tuning de Modelos Pré-treinados

1. Use técnicas de Parameter-Efficient Fine-Tuning (PEFT) como LoRA ou Adaptadores
2. Congele a maioria dos parâmetros do modelo
3. Use offloading para CPU quando necessário
4. Considere quantização de 8 bits para os parâmetros congelados

### Para Inferência

1. Quantize o modelo para INT8 ou menor
2. Use geração em lotes pequenos
3. Considere modelos destilados ou pruned
4. Implemente técnicas de cache para tokens já processados

## Exemplo Prático: Configurando um Ambiente Otimizado

Vamos concluir com um exemplo prático de configuração para treinar um modelo de linguagem pequeno em uma GPU com 6GB de memória:

```python
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from accelerate import Accelerator

# 1. Configurar acelerador com mixed precision
accelerator = Accelerator(mixed_precision='fp16')

# 2. Carregar modelo pequeno (GPT-2 pequeno tem ~124M parâmetros)
model_name = "gpt2"  # ~124M parâmetros
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Habilitar gradient checkpointing
model.gradient_checkpointing_enable()

# 4. Preparar dataset (exemplo com um dataset pequeno)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples["text"], truncation=True, max_length=128),
    batched=True,
    remove_columns=["text"]
)

# 5. Configurar treinamento com batch size pequeno e acumulação de gradientes
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Simula batch size de 8
    learning_rate=5e-5,
    num_train_epochs=3,
    save_strategy="epoch",
    fp16=True,  # Usar mixed precision
    report_to="none",  # Desativar relatórios para economizar memória
)

# 6. Preparar trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# 7. Treinar o modelo
trainer.train()

# 8. Salvar o modelo
model.save_pretrained("./modelo_otimizado")
tokenizer.save_pretrained("./modelo_otimizado")
```

## Conclusão

Trabalhar com LLMs em hardware com limitações de memória é desafiador, mas totalmente viável com as técnicas de otimização adequadas. Ao combinar estratégias como redução de precisão, gradient checkpointing, offloading e acumulação de gradientes, podemos construir, treinar e utilizar modelos surpreendentemente poderosos mesmo em GPUs com 4-6GB de memória ou no Google Colab.

Na próxima aula, exploraremos os fundamentos matemáticos e teóricos por trás dos LLMs, estabelecendo a base conceitual necessária para implementar esses modelos do zero.

## Referências e Leituras Adicionais

1. Rajbhandari, S., et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models".
2. Kitaev, N., et al. (2020). "Reformer: The Efficient Transformer".
3. Wang, S., et al. (2020). "Linformer: Self-Attention with Linear Complexity".
4. Dettmers, T., et al. (2022). "8-bit Optimizers via Block-wise Quantization".
5. Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models".
