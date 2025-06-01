# Configuração do Ambiente de Desenvolvimento

Neste módulo, vamos preparar todo o ambiente necessário para desenvolver, treinar e avaliar nosso LLM. Uma configuração adequada é fundamental para garantir que possamos trabalhar eficientemente com recursos computacionais limitados, como GPUs com 4-6GB de memória ou o Google Colab.

## Objetivos do Módulo

Ao final deste módulo, você será capaz de:

1. Configurar um ambiente de desenvolvimento completo para LLMs
2. Instalar e configurar todas as dependências necessárias
3. Configurar corretamente o PyTorch para utilizar GPU com eficiência
4. Monitorar e gerenciar recursos computacionais durante o desenvolvimento
5. Preparar ambientes alternativos (Google Colab) para casos de limitação de hardware

## Instalação de Dependências

Vamos começar instalando as bibliotecas essenciais para o desenvolvimento de LLMs. Utilizaremos principalmente o PyTorch como framework de deep learning, junto com bibliotecas auxiliares para processamento de texto e otimização.

### Ambiente Local

Se você estiver trabalhando em sua máquina local, recomendamos a criação de um ambiente virtual para isolar as dependências:

```bash
# Criar ambiente virtual
python -m venv llm_env

# Ativar o ambiente (Windows)
llm_env\Scripts\activate

# Ativar o ambiente (Linux/Mac)
source llm_env/bin/activate
```

Agora, vamos instalar as dependências principais:

```bash
# Instalar PyTorch com suporte a CUDA
# Nota: A versão exata pode variar dependendo da sua GPU e sistema
# Verifique a versão compatível em https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Bibliotecas para processamento de texto e NLP
pip install transformers datasets tokenizers sentencepiece

# Bibliotecas para otimização e eficiência
pip install accelerate bitsandbytes deepspeed

# Bibliotecas para visualização e monitoramento
pip install matplotlib tensorboard wandb

# Utilitários
pip install tqdm numpy pandas scikit-learn
```

### Google Colab

Se você estiver utilizando o Google Colab, muitas dessas bibliotecas já vêm pré-instaladas. No entanto, algumas versões podem estar desatualizadas ou faltando. Aqui está um script para garantir que tudo esteja instalado corretamente:

```python
# Verificar a GPU disponível
!nvidia-smi

# Instalar ou atualizar bibliotecas
!pip install -q transformers datasets tokenizers sentencepiece
!pip install -q accelerate bitsandbytes
!pip install -q wandb

# Montar o Google Drive para armazenamento persistente
from google.colab import drive
drive.mount('/content/drive')

# Criar diretório para o projeto
!mkdir -p /content/drive/MyDrive/llm_project
```

## Verificação da Configuração de GPU

É essencial verificar se o PyTorch está reconhecendo corretamente sua GPU e se a configuração CUDA está funcionando:

```python
import torch

# Verificar se CUDA está disponível
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo atual: {torch.cuda.get_device_name(0)}")
    print(f"Número de GPUs disponíveis: {torch.cuda.device_count()}")
    print(f"Memória total da GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Memória alocada: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Memória reservada: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
```

## Otimização da Configuração para GPUs com Memória Limitada

Para GPUs com 4-6GB de memória, precisamos aplicar algumas otimizações específicas:

```python
import os
import torch

# Limitar cache CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Habilitar TF32 para operações de matriz (apenas para GPUs Ampere ou mais recentes)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configurar para liberar memória de cache não utilizada
torch.cuda.empty_cache()

# Função para monitorar uso de memória
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"Memória alocada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Memória reservada: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        # Liberar cache não utilizado
        torch.cuda.empty_cache()
        print(f"Após limpeza - Memória reservada: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Configuração do Ambiente para Treinamento Eficiente

Vamos configurar o ambiente para treinamento com precisão mista, que é essencial para economizar memória:

```python
import torch
from accelerate import Accelerator

# Configurar acelerador com precisão mista
accelerator = Accelerator(mixed_precision='fp16')

# Exemplo de uso com um modelo, otimizador e dataloader
def setup_training(model, optimizer, train_dataloader):
    # Preparar para treinamento distribuído e precisão mista
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    return model, optimizer, train_dataloader

# Exemplo de loop de treinamento com acumulação de gradientes
def train_with_gradient_accumulation(model, optimizer, train_dataloader, num_epochs, accumulation_steps=8):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_dataloader):
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
            
            # Backward pass com acelerador
            accelerator.backward(loss)
            
            # Atualizar parâmetros a cada accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader):.4f}")
```

## Ferramentas de Monitoramento de Recursos

É crucial monitorar o uso de recursos durante o desenvolvimento e treinamento de LLMs. Vamos configurar algumas ferramentas para isso:

### TensorBoard para Monitoramento Local

```python
from torch.utils.tensorboard import SummaryWriter

# Inicializar o writer
writer = SummaryWriter(log_dir='./logs')

# Exemplo de uso durante treinamento
def log_metrics(writer, loss, step, prefix='train'):
    writer.add_scalar(f'{prefix}/loss', loss, step)
    
    # Também podemos registrar uso de memória
    if torch.cuda.is_available():
        writer.add_scalar(f'{prefix}/gpu_memory_allocated', 
                         torch.cuda.memory_allocated() / 1e9, step)

# Não esqueça de fechar o writer ao final
# writer.close()
```

### Weights & Biases para Experimentos em Nuvem

```python
import wandb

# Inicializar o projeto
wandb.init(project="llm-from-scratch", name="training-run-1")

# Configurar hiperparâmetros para rastreamento
config = wandb.config
config.learning_rate = 5e-5
config.batch_size = 4
config.accumulation_steps = 8
config.model_size = "small"

# Exemplo de log durante treinamento
def log_to_wandb(loss, step):
    wandb.log({
        "loss": loss,
        "gpu_memory": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    }, step=step)

# Finalizar ao terminar
# wandb.finish()
```

## Estrutura de Diretórios do Projeto

Vamos organizar nosso projeto com uma estrutura clara:

```
llm_project/
├── data/                  # Dados para treinamento e avaliação
│   ├── raw/               # Dados brutos
│   └── processed/         # Dados processados
├── models/                # Definições de modelos
│   ├── tokenizer/         # Tokenizadores
│   ├── transformer/       # Componentes do transformer
│   └── utils/             # Utilitários para modelos
├── training/              # Scripts de treinamento
│   ├── configs/           # Configurações de treinamento
│   └── callbacks/         # Callbacks para monitoramento
├── evaluation/            # Scripts de avaliação
├── inference/             # Scripts para inferência
├── notebooks/             # Jupyter notebooks para experimentos
├── checkpoints/           # Pontos de salvamento de modelos
└── logs/                  # Logs de treinamento
```

Vamos criar esta estrutura:

```python
import os

def create_project_structure(base_dir):
    """Cria a estrutura de diretórios para o projeto."""
    directories = [
        'data/raw',
        'data/processed',
        'models/tokenizer',
        'models/transformer',
        'models/utils',
        'training/configs',
        'training/callbacks',
        'evaluation',
        'inference',
        'notebooks',
        'checkpoints',
        'logs'
    ]
    
    for directory in directories:
        path = os.path.join(base_dir, directory)
        os.makedirs(path, exist_ok=True)
        print(f"Criado: {path}")

# Exemplo de uso
# Para ambiente local
create_project_structure('./llm_project')

# Para Google Colab
# create_project_structure('/content/drive/MyDrive/llm_project')
```

## Configuração de Ambiente para Desenvolvimento Colaborativo

Se você estiver trabalhando em equipe ou quiser manter seu código versionado, é recomendável usar o Git:

```bash
# Inicializar repositório Git
git init

# Criar arquivo .gitignore para excluir arquivos desnecessários
cat > .gitignore << EOL
# Ambientes virtuais
venv/
env/
llm_env/

# Arquivos de cache
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/

# Logs e checkpoints
logs/
checkpoints/
wandb/

# Dados grandes
data/raw/
data/processed/

# Configurações locais
.env
EOL

# Commit inicial
git add .
git commit -m "Configuração inicial do projeto"
```

## Verificação Final do Ambiente

Vamos criar um script para verificar se tudo está configurado corretamente:

```python
def check_environment():
    """Verifica se o ambiente está configurado corretamente."""
    import sys
    import torch
    import transformers
    import datasets
    import accelerate
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Datasets version: {datasets.__version__}")
    print(f"Accelerate version: {accelerate.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Testar operações básicas na GPU
        a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
        print(f"GPU test: {a + b}")
        print("GPU está funcionando corretamente!")
    else:
        print("AVISO: CUDA não está disponível. O treinamento será muito lento na CPU.")
    
    # Verificar se podemos carregar um modelo pequeno
    try:
        from transformers import AutoTokenizer, AutoModel
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # Testar tokenização e inferência
        inputs = tokenizer("Testando o ambiente de LLM", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            model = model.to('cuda')
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("Modelo carregado e testado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
    
    print("\nVerificação de ambiente concluída!")

# Executar verificação
check_environment()
```

## Conclusão

Nesta aula, configuramos um ambiente de desenvolvimento completo para trabalhar com LLMs, otimizado para hardware com limitações de memória. Instalamos todas as dependências necessárias, configuramos ferramentas de monitoramento e criamos uma estrutura organizada para nosso projeto.

Na próxima aula, vamos explorar a manipulação de dados textuais, incluindo técnicas de coleta, limpeza e pré-processamento, que são fundamentais para o treinamento eficaz de modelos de linguagem.

## Exercícios Práticos

1. Configure o ambiente em sua máquina local ou no Google Colab seguindo as instruções desta aula.
2. Execute o script de verificação e resolva quaisquer problemas encontrados.
3. Experimente com diferentes configurações de memória CUDA e observe o impacto no uso de recursos.
4. Crie um notebook Jupyter para testar o carregamento de um modelo pequeno (como GPT-2 small) e verifique o consumo de memória.
5. Configure o TensorBoard ou Weights & Biases e pratique o registro de métricas simples.
