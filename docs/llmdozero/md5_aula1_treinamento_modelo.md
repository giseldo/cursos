# Treinamento do Modelo

Neste módulo, vamos explorar o processo de treinamento de um LLM de forma eficiente, mesmo com recursos computacionais limitados. Aprenderemos a configurar hiperparâmetros, implementar estratégias de otimização e monitorar o treinamento para obter os melhores resultados possíveis.

## Preparação para Treinamento

Antes de iniciar o treinamento propriamente dito, precisamos configurar adequadamente o ambiente e os hiperparâmetros.

### Definição de Hiperparâmetros

Os hiperparâmetros são cruciais para o sucesso do treinamento. Vamos explorar os principais hiperparâmetros e como escolhê-los:

```python
class TrainingConfig:
    """
    Configuração para treinamento de LLM.
    """
    def __init__(
        self,
        # Parâmetros do modelo
        vocab_size=50257,
        hidden_size=384,
        num_layers=6,
        num_heads=6,
        intermediate_size=1536,
        max_position_embeddings=1024,
        
        # Parâmetros de treinamento
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        num_epochs=3,
        warmup_steps=500,
        
        # Otimizações de memória
        fp16=True,
        gradient_checkpointing=True,
        cpu_offload=False,
        
        # Parâmetros de avaliação
        eval_steps=500,
        save_steps=1000,
        
        # Diretórios
        output_dir="./output",
        logging_dir="./logs",
    ):
        """
        Inicializa a configuração de treinamento.
        
        Args:
            vocab_size: Tamanho do vocabulário
            hidden_size: Dimensão dos embeddings e camadas ocultas
            num_layers: Número de camadas do Transformer
            num_heads: Número de cabeças de atenção
            intermediate_size: Dimensão da camada feed-forward
            max_position_embeddings: Número máximo de posições para embeddings posicionais
            
            batch_size: Tamanho do batch
            gradient_accumulation_steps: Número de passos para acumulação de gradientes
            learning_rate: Taxa de aprendizado
            weight_decay: Decaimento de peso para regularização
            adam_beta1: Beta1 para otimizador Adam
            adam_beta2: Beta2 para otimizador Adam
            adam_epsilon: Epsilon para otimizador Adam
            max_grad_norm: Norma máxima para clipping de gradiente
            num_epochs: Número de épocas
            warmup_steps: Número de passos de warmup
            
            fp16: Se True, usa precisão mista (FP16)
            gradient_checkpointing: Se True, usa gradient checkpointing
            cpu_offload: Se True, aplica offloading para CPU
            
            eval_steps: Número de passos entre avaliações
            save_steps: Número de passos entre salvamentos do modelo
            
            output_dir: Diretório para salvar o modelo
            logging_dir: Diretório para logs
        """
        # Atribuir todos os parâmetros como atributos
        for key, value in locals().items():
            if key != 'self':
                setattr(self, key, value)
        
        # Calcular batch size efetivo
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        
        # Verificar compatibilidade de parâmetros
        assert hidden_size % num_heads == 0, "hidden_size deve ser divisível por num_heads"
    
    def to_dict(self):
        """
        Converte a configuração para um dicionário.
        
        Returns:
            Dict: Configuração como dicionário
        """
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict):
        """
        Cria uma configuração a partir de um dicionário.
        
        Args:
            config_dict: Dicionário de configuração
            
        Returns:
            TrainingConfig: Configuração criada
        """
        return cls(**config_dict)
    
    def save(self, filepath):
        """
        Salva a configuração em um arquivo JSON.
        
        Args:
            filepath: Caminho para o arquivo
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """
        Carrega uma configuração de um arquivo JSON.
        
        Args:
            filepath: Caminho para o arquivo
            
        Returns:
            TrainingConfig: Configuração carregada
        """
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
```

### Escolha de Hiperparâmetros para Hardware Limitado

Quando trabalhamos com GPUs de 4-6GB ou no Google Colab, precisamos ser especialmente cuidadosos com a escolha de hiperparâmetros:

```python
def get_memory_optimized_config(gpu_memory_gb=6):
    """
    Retorna uma configuração otimizada para uma GPU com memória limitada.
    
    Args:
        gpu_memory_gb: Memória da GPU em GB
        
    Returns:
        TrainingConfig: Configuração otimizada
    """
    if gpu_memory_gb <= 4:
        # Configuração para GPUs com 4GB ou menos
        return TrainingConfig(
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            intermediate_size=1024,
            batch_size=1,
            gradient_accumulation_steps=16,
            fp16=True,
            gradient_checkpointing=True,
            max_position_embeddings=512,
        )
    elif gpu_memory_gb <= 6:
        # Configuração para GPUs com 6GB
        return TrainingConfig(
            hidden_size=384,
            num_layers=6,
            num_heads=6,
            intermediate_size=1536,
            batch_size=2,
            gradient_accumulation_steps=8,
            fp16=True,
            gradient_checkpointing=True,
            max_position_embeddings=1024,
        )
    elif gpu_memory_gb <= 8:
        # Configuração para GPUs com 8GB
        return TrainingConfig(
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            intermediate_size=2048,
            batch_size=4,
            gradient_accumulation_steps=4,
            fp16=True,
            gradient_checkpointing=True,
            max_position_embeddings=1024,
        )
    else:
        # Configuração para GPUs com mais de 8GB
        return TrainingConfig(
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            intermediate_size=3072,
            batch_size=8,
            gradient_accumulation_steps=2,
            fp16=True,
            gradient_checkpointing=False,
            max_position_embeddings=2048,
        )
```

### Estratégias de Otimização

A escolha do otimizador e do scheduler é fundamental para o treinamento eficiente:

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def get_optimizer_and_scheduler(model, config, num_training_steps):
    """
    Configura o otimizador e o scheduler para treinamento.
    
    Args:
        model: Modelo a ser treinado
        config: Configuração de treinamento
        num_training_steps: Número total de passos de treinamento
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Separar parâmetros com e sem weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Criar otimizador
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
    
    # Criar scheduler com warmup
    def lr_lambda(current_step):
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - config.warmup_steps))
        )
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler
```

### Configuração de Checkpoints

É importante salvar checkpoints regularmente durante o treinamento para evitar perda de progresso:

```python
import os
import torch

class CheckpointManager:
    """
    Gerencia checkpoints durante o treinamento.
    """
    def __init__(self, output_dir, model_name="model", save_steps=1000, max_checkpoints=3):
        """
        Inicializa o gerenciador de checkpoints.
        
        Args:
            output_dir: Diretório para salvar checkpoints
            model_name: Nome base para os arquivos de checkpoint
            save_steps: Número de passos entre salvamentos
            max_checkpoints: Número máximo de checkpoints a manter
        """
        self.output_dir = output_dir
        self.model_name = model_name
        self.save_steps = save_steps
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, scheduler, global_step, epoch, loss, config):
        """
        Salva um checkpoint do modelo.
        
        Args:
            model: Modelo a ser salvo
            optimizer: Otimizador
            scheduler: Scheduler
            global_step: Passo global atual
            epoch: Época atual
            loss: Perda atual
            config: Configuração de treinamento
        """
        # Verificar se é hora de salvar
        if global_step % self.save_steps != 0:
            return
        
        # Criar nome do checkpoint
        checkpoint_name = f"{self.model_name}_step_{global_step}"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        
        # Criar diretório para o checkpoint
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Salvar modelo
        model.save_pretrained(checkpoint_path)
        
        # Salvar estado do treinamento
        training_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "loss": loss,
        }
        torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"))
        
        # Salvar configuração
        config.save(os.path.join(checkpoint_path, "config.json"))
        
        # Adicionar à lista de checkpoints
        self.checkpoints.append(checkpoint_path)
        
        # Remover checkpoints antigos se exceder o máximo
        if len(self.checkpoints) > self.max_checkpoints:
            oldest_checkpoint = self.checkpoints.pop(0)
            try:
                import shutil
                shutil.rmtree(oldest_checkpoint)
                print(f"Removido checkpoint antigo: {oldest_checkpoint}")
            except Exception as e:
                print(f"Erro ao remover checkpoint antigo: {e}")
    
    def load_latest_checkpoint(self, model, optimizer=None, scheduler=None):
        """
        Carrega o checkpoint mais recente.
        
        Args:
            model: Modelo a ser carregado
            optimizer: Otimizador (opcional)
            scheduler: Scheduler (opcional)
            
        Returns:
            tuple: (global_step, epoch, loss) ou None se não houver checkpoint
        """
        # Encontrar checkpoints existentes
        checkpoints = []
        for dirname in os.listdir(self.output_dir):
            if dirname.startswith(self.model_name) and "step_" in dirname:
                checkpoint_path = os.path.join(self.output_dir, dirname)
                step = int(dirname.split("step_")[1])
                checkpoints.append((step, checkpoint_path))
        
        if not checkpoints:
            return None
        
        # Ordenar por passo e pegar o mais recente
        checkpoints.sort(key=lambda x: x[0])
        _, latest_checkpoint = checkpoints[-1]
        
        # Atualizar lista de checkpoints
        self.checkpoints = [cp[1] for cp in checkpoints]
        
        # Carregar modelo
        model.load_pretrained(latest_checkpoint)
        
        # Carregar estado do treinamento
        training_state_path = os.path.join(latest_checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            
            if optimizer is not None:
                optimizer.load_state_dict(training_state["optimizer"])
            
            if scheduler is not None:
                scheduler.load_state_dict(training_state["scheduler"])
            
            return (
                training_state["global_step"],
                training_state["epoch"],
                training_state["loss"],
            )
        
        return None
```

## Treinamento Eficiente

Agora, vamos implementar o loop de treinamento com todas as otimizações necessárias para hardware limitado.

### Implementação do Loop de Treinamento

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import math
from torch.cuda.amp import autocast, GradScaler

def train_model(model, train_dataset, config, eval_dataset=None):
    """
    Treina um modelo LLM com otimizações para hardware limitado.
    
    Args:
        model: Modelo a ser treinado
        train_dataset: Dataset de treinamento
        config: Configuração de treinamento
        eval_dataset: Dataset de avaliação (opcional)
        
    Returns:
        Modelo treinado
    """
    # Configurar device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mover modelo para o device
    model.to(device)
    
    # Habilitar gradient checkpointing se configurado
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Configurar dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    # Calcular número total de passos de treinamento
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    num_training_steps = config.num_epochs * num_update_steps_per_epoch
    
    # Configurar otimizador e scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, config, num_training_steps)
    
    # Configurar scaler para precisão mista
    scaler = GradScaler() if config.fp16 else None
    
    # Configurar gerenciador de checkpoints
    checkpoint_manager = CheckpointManager(
        output_dir=config.output_dir,
        save_steps=config.save_steps,
    )
    
    # Tentar carregar checkpoint existente
    resume_info = checkpoint_manager.load_latest_checkpoint(model, optimizer, scheduler)
    if resume_info is not None:
        global_step, start_epoch, _ = resume_info
        print(f"Retomando treinamento do passo {global_step}, época {start_epoch}")
    else:
        global_step = 0
        start_epoch = 0
    
    # Configurar TensorBoard para logging
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=config.logging_dir)
    
    # Função para calcular perda
    def compute_loss(logits, labels):
        # Deslocar logits e labels para cálculo da perda
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calcular perda de entropia cruzada
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
    
    # Loop de treinamento
    print("Iniciando treinamento...")
    model.train()
    
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Época {epoch+1}/{config.num_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            # Extrair input_ids e attention_mask
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass com precisão mista se configurado
            if config.fp16:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                    loss = compute_loss(logits, input_ids)
                    loss = loss / config.gradient_accumulation_steps
                
                # Backward pass com scaler
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                loss = compute_loss(logits, input_ids)
                loss = loss / config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            # Atualizar parâmetros a cada gradient_accumulation_steps
            if (step + 1) % config.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # Clipping de gradiente
                if config.fp16:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Atualizar parâmetros
                if config.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if global_step % 10 == 0:
                    writer.add_scalar("train/loss", loss.item() * config.gradient_accumulation_steps, global_step)
                    writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
                
                # Avaliação
                if eval_dataset is not None and global_step % config.eval_steps == 0:
                    eval_loss = evaluate_model(model, eval_dataset, device, config)
                    writer.add_scalar("eval/loss", eval_loss, global_step)
                    
                    print(f"\nPasso {global_step}: Perda de avaliação: {eval_loss:.4f}")
                    
                    # Voltar para modo de treinamento
                    model.train()
                
                # Salvar checkpoint
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch,
                    loss=total_loss / (step + 1),
                    config=config,
                )
            
            # Atualizar barra de progresso
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": total_loss / (step + 1)})
            
            # Liberar memória
            torch.cuda.empty_cache()
        
        # Estatísticas da época
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_dataloader)
        
        print(f"Época {epoch+1} concluída em {epoch_time:.2f}s. Perda média: {avg_loss:.4f}")
    
    # Fechar writer
    writer.close()
    
    # Salvar modelo final
    final_model_path = os.path.join(config.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    
    return model

def evaluate_model(model, eval_dataset, device, config):
    """
    Avalia o modelo em um dataset de avaliação.
    
    Args:
        model: Modelo a ser avaliado
        eval_dataset: Dataset de avaliação
        device: Dispositivo para avaliação
        config: Configuração de treinamento
        
    Returns:
        float: Perda média de avaliação
    """
    model.eval()
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    
    total_loss = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Extrair input_ids e attention_mask
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass com precisão mista se configurado
            if config.fp16:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                    
                    # Calcular perda
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                
                # Calcular perda
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            
            total_loss += loss.item()
    
    return total_loss / len(eval_dataloader)
```

### Técnicas de Acumulação de Gradientes

A acumulação de gradientes é essencial para treinar com batches efetivamente maiores em hardware limitado:

```python
def demonstrate_gradient_accumulation():
    """
    Demonstra como a acumulação de gradientes funciona.
    """
    import torch
    import torch.nn as nn
    
    # Modelo simples para demonstração
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Dados de exemplo
    batch_size = 4
    num_batches = 4
    accumulation_steps = 4
    
    # Gerar dados aleatórios
    all_data = torch.randn(batch_size * num_batches, 10)
    all_targets = torch.randn(batch_size * num_batches, 1)
    
    # Abordagem 1: Batch grande (referência)
    model.zero_grad()
    
    # Forward pass com todos os dados
    outputs = model(all_data)
    loss = nn.MSELoss()(outputs, all_targets)
    
    # Backward pass
    loss.backward()
    
    # Guardar gradientes para comparação
    reference_grads = {}
    for name, param in model.named_parameters():
        reference_grads[name] = param.grad.clone()
    
    # Resetar gradientes
    optimizer.zero_grad()
    
    # Abordagem 2: Acumulação de gradientes
    for i in range(num_batches):
        # Extrair mini-batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        data = all_data[start_idx:end_idx]
        targets = all_targets[start_idx:end_idx]
        
        # Forward pass
        outputs = model(data)
        loss = nn.MSELoss()(outputs, targets)
        
        # Normalizar perda pelo número de passos de acumulação
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Atualizar parâmetros apenas no último passo de acumulação
        if (i + 1) % accumulation_steps == 0:
            # Aqui faríamos optimizer.step() em um treinamento real
            pass
    
    # Comparar gradientes
    for name, param in model.named_parameters():
        accumulated_grad = param.grad
        reference_grad = reference_grads[name]
        
        # Calcular diferença relativa
        diff = torch.norm(accumulated_grad - reference_grad) / torch.norm(reference_grad)
        
        print(f"Parâmetro: {name}")
        print(f"  Gradiente de referência: {reference_grad.mean().item():.6f}")
        print(f"  Gradiente acumulado: {accumulated_grad.mean().item():.6f}")
        print(f"  Diferença relativa: {diff.item():.6f}")
        
    print("\nConclusão: A acumulação de gradientes produz resultados equivalentes a usar um batch maior,")
    print("mas com requisitos de memória muito menores, pois processamos mini-batches sequencialmente.")
```

### Monitoramento e Debugging

O monitoramento adequado é crucial para identificar e resolver problemas durante o treinamento:

```python
def setup_monitoring(config):
    """
    Configura ferramentas de monitoramento para o treinamento.
    
    Args:
        config: Configuração de treinamento
        
    Returns:
        dict: Ferramentas de monitoramento
    """
    import os
    from torch.utils.tensorboard import SummaryWriter
    
    # Criar diretórios
    os.makedirs(config.logging_dir, exist_ok=True)
    
    # Configurar TensorBoard
    writer = SummaryWriter(log_dir=config.logging_dir)
    
    # Função para monitorar uso de GPU
    def log_gpu_usage(step):
        if torch.cuda.is_available():
            # Memória alocada
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            writer.add_scalar("system/gpu_memory_allocated_gb", allocated, step)
            
            # Memória reservada
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            writer.add_scalar("system/gpu_memory_reserved_gb", reserved, step)
            
            # Utilização
            if hasattr(torch.cuda, "utilization"):
                utilization = torch.cuda.utilization() / 100.0
                writer.add_scalar("system/gpu_utilization", utilization, step)
    
    # Função para registrar gradientes
    def log_gradients(model, step):
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                writer.add_histogram(f"gradients/{name}", param.grad, step)
    
    # Função para registrar pesos
    def log_weights(model, step):
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f"weights/{name}", param, step)
    
    # Função para registrar perda
    def log_loss(loss, step, prefix="train"):
        writer.add_scalar(f"{prefix}/loss", loss, step)
    
    # Função para registrar taxa de aprendizado
    def log_learning_rate(scheduler, step):
        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], step)
    
    # Função para registrar perplexidade
    def log_perplexity(loss, step, prefix="train"):
        perplexity = torch.exp(torch.tensor(loss))
        writer.add_scalar(f"{prefix}/perplexity", perplexity, step)
    
    # Função para registrar exemplos de geração
    def log_generation_samples(model, tokenizer, prompts, step, max_length=50, num_return_sequences=1):
        model.eval()
        
        for i, prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            
            # Gerar texto
            with torch.no_grad():
                output_sequences = model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                )
            
            # Decodificar e registrar
            for j, sequence in enumerate(output_sequences):
                generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
                writer.add_text(f"generation/prompt_{i}_sample_{j}", generated_text, step)
        
        model.train()
    
    # Retornar funções de monitoramento
    return {
        "writer": writer,
        "log_gpu_usage": log_gpu_usage,
        "log_gradients": log_gradients,
        "log_weights": log_weights,
        "log_loss": log_loss,
        "log_learning_rate": log_learning_rate,
        "log_perplexity": log_perplexity,
        "log_generation_samples": log_generation_samples,
    }
```

## Técnicas Avançadas de Treinamento

Além das técnicas básicas, existem abordagens avançadas que podem melhorar significativamente o treinamento de LLMs.

### Curriculum Learning

O curriculum learning envolve treinar o modelo em dados progressivamente mais difíceis:

```python
class CurriculumSampler:
    """
    Implementa curriculum learning para treinamento de LLM.
    """
    def __init__(self, datasets, difficulties, schedule="linear", total_steps=10000):
        """
        Inicializa o amostrador de curriculum.
        
        Args:
            datasets: Lista de datasets, do mais fácil para o mais difícil
            difficulties: Lista de valores de dificuldade para cada dataset
            schedule: Tipo de agendamento ("linear", "exp", "step")
            total_steps: Número total de passos de treinamento
        """
        self.datasets = datasets
        self.difficulties = difficulties
        self.schedule = schedule
        self.total_steps = total_steps
        
        # Verificar se os tamanhos correspondem
        assert len(datasets) == len(difficulties), "Número de datasets e dificuldades deve ser igual"
        
        # Normalizar dificuldades
        self.difficulties = [d / max(difficulties) for d in difficulties]
        
        # Calcular tamanhos dos datasets
        self.dataset_sizes = [len(ds) for ds in datasets]
        self.total_size = sum(self.dataset_sizes)
    
    def get_curriculum_weights(self, step):
        """
        Calcula os pesos para cada dataset com base no passo atual.
        
        Args:
            step: Passo atual de treinamento
            
        Returns:
            list: Pesos para cada dataset
        """
        # Calcular progresso (0 a 1)
        progress = min(1.0, step / self.total_steps)
        
        if self.schedule == "linear":
            # Agendamento linear
            weights = [max(0, 1.0 - abs(progress - d) * 2) for d in self.difficulties]
        elif self.schedule == "exp":
            # Agendamento exponencial
            weights = [math.exp(-5 * abs(progress - d)) for d in self.difficulties]
        elif self.schedule == "step":
            # Agendamento em etapas
            segment_size = 1.0 / len(self.datasets)
            weights = [0] * len(self.datasets)
            current_segment = int(progress / segment_size)
            if current_segment < len(weights):
                weights[current_segment] = 1.0
        else:
            raise ValueError(f"Agendamento desconhecido: {self.schedule}")
        
        # Normalizar pesos
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Fallback para distribuição uniforme
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights
    
    def sample_batch(self, step, batch_size):
        """
        Amostra um batch com base no curriculum.
        
        Args:
            step: Passo atual de treinamento
            batch_size: Tamanho do batch
            
        Returns:
            dict: Batch amostrado
        """
        # Obter pesos do curriculum
        weights = self.get_curriculum_weights(step)
        
        # Determinar número de exemplos de cada dataset
        counts = [0] * len(self.datasets)
        for _ in range(batch_size):
            # Escolher dataset com base nos pesos
            dataset_idx = random.choices(range(len(self.datasets)), weights=weights)[0]
            counts[dataset_idx] += 1
        
        # Amostrar de cada dataset
        batch = {"input_ids": [], "attention_mask": []}
        
        for i, count in enumerate(counts):
            if count > 0:
                # Índices aleatórios para este dataset
                indices = random.sample(range(len(self.datasets[i])), count)
                
                # Obter exemplos
                for idx in indices:
                    example = self.datasets[i][idx]
                    for key in example:
                        if key not in batch:
                            batch[key] = []
                        batch[key].append(example[key])
        
        # Converter listas para tensores
        for key in batch:
            batch[key] = torch.stack(batch[key])
        
        return batch
```

### Mixed-Precision Training

O treinamento com precisão mista é crucial para economizar memória:

```python
def explain_mixed_precision():
    """
    Explica como o treinamento com precisão mista funciona.
    """
    explanation = """
    # Treinamento com Precisão Mista (FP16)
    
    O treinamento com precisão mista usa FP16 (16 bits) para a maioria das operações, 
    mas mantém algumas operações críticas em FP32 (32 bits) para estabilidade.
    
    ## Benefícios:
    
    1. **Economia de Memória**: Reduz o consumo de memória pela metade para a maioria dos tensores.
    2. **Aceleração**: GPUs modernas têm hardware especializado para operações FP16, resultando em treinamento mais rápido.
    3. **Largura de Banda**: Transferência de dados mais eficiente entre memória e GPU.
    
    ## Como Funciona:
    
    1. **Forward Pass**: Realizado em FP16 para economizar memória.
    2. **Gradientes**: Calculados em FP16 durante o backward pass.
    3. **Atualização de Pesos**: Convertida para FP32 para evitar problemas de precisão.
    4. **Scaling de Perda**: Usado para evitar underflow em FP16.
    
    ## Implementação com PyTorch:
    
    ```python
    # Importar módulos necessários
    from torch.cuda.amp import autocast, GradScaler
    
    # Inicializar scaler
    scaler = GradScaler()
    
    # Loop de treinamento
    for batch in dataloader:
        # Forward pass em FP16
        with autocast():
            outputs = model(batch)
            loss = loss_fn(outputs, targets)
        
        # Backward pass com scaling
        scaler.scale(loss).backward()
        
        # Unscale gradientes e aplicar clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Atualizar pesos com gradientes unscaled
        scaler.step(optimizer)
        
        # Atualizar fator de escala para próxima iteração
        scaler.update()
        
        # Zerar gradientes
        optimizer.zero_grad()
    ```
    
    ## Considerações:
    
    - Nem todas as operações são estáveis em FP16 (ex: softmax, normalização de camada)
    - PyTorch lida automaticamente com essas operações críticas em FP32
    - O GradScaler ajusta dinamicamente o fator de escala para evitar underflow
    - Algumas arquiteturas podem requerer ajustes específicos para estabilidade
    """
    
    return explanation
```

### Distilação de Conhecimento

A distilação permite treinar modelos menores a partir de modelos maiores:

```python
def knowledge_distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """
    Calcula a perda de distilação de conhecimento.
    
    Args:
        student_logits: Logits do modelo estudante
        teacher_logits: Logits do modelo professor
        labels: Labels verdadeiros
        temperature: Temperatura para softmax
        alpha: Peso para balancear perda de distilação e perda de tarefa
        
    Returns:
        torch.Tensor: Perda combinada
    """
    # Perda de distilação (KL divergence)
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    distillation_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    
    # Perda de tarefa (cross entropy)
    task_loss = F.cross_entropy(student_logits, labels)
    
    # Combinar perdas
    combined_loss = alpha * distillation_loss + (1 - alpha) * task_loss
    
    return combined_loss

def train_with_distillation(student_model, teacher_model, train_dataset, config):
    """
    Treina um modelo estudante usando distilação de conhecimento.
    
    Args:
        student_model: Modelo estudante a ser treinado
        teacher_model: Modelo professor pré-treinado
        train_dataset: Dataset de treinamento
        config: Configuração de treinamento
        
    Returns:
        Modelo estudante treinado
    """
    # Configurar device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mover modelos para o device
    student_model.to(device)
    teacher_model.to(device)
    
    # Colocar professor em modo de avaliação
    teacher_model.eval()
    
    # Configurar dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    # Calcular número total de passos de treinamento
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    num_training_steps = config.num_epochs * num_update_steps_per_epoch
    
    # Configurar otimizador e scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(student_model, config, num_training_steps)
    
    # Configurar scaler para precisão mista
    scaler = GradScaler() if config.fp16 else None
    
    # Loop de treinamento
    print("Iniciando treinamento com distilação...")
    student_model.train()
    
    global_step = 0
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Época {epoch+1}/{config.num_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            # Extrair input_ids e attention_mask
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass com precisão mista se configurado
            if config.fp16:
                with autocast():
                    # Forward pass do estudante
                    student_outputs = student_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    student_logits = student_outputs["logits"] if isinstance(student_outputs, dict) else student_outputs[0]
                    
                    # Forward pass do professor (sem gradientes)
                    with torch.no_grad():
                        teacher_outputs = teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        teacher_logits = teacher_outputs["logits"] if isinstance(teacher_outputs, dict) else teacher_outputs[0]
                    
                    # Calcular perda de distilação
                    # Para simplificar, usamos os próprios input_ids como labels (próximo token)
                    shift_logits_student = student_logits[..., :-1, :].contiguous()
                    shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    # Calcular perda para cada posição
                    loss = 0
                    for pos in range(shift_labels.size(1)):
                        pos_student_logits = shift_logits_student[:, pos, :]
                        pos_teacher_logits = shift_logits_teacher[:, pos, :]
                        pos_labels = shift_labels[:, pos]
                        
                        pos_loss = knowledge_distillation_loss(
                            pos_student_logits,
                            pos_teacher_logits,
                            pos_labels,
                            temperature=2.0,
                            alpha=0.5,
                        )
                        loss += pos_loss
                    
                    # Normalizar perda pelo número de posições
                    loss = loss / shift_labels.size(1)
                    
                    # Normalizar perda pelo número de passos de acumulação
                    loss = loss / config.gradient_accumulation_steps
                
                # Backward pass com scaler
                scaler.scale(loss).backward()
            else:
                # Implementação sem precisão mista (similar à acima)
                # ...
                pass
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            # Atualizar parâmetros a cada gradient_accumulation_steps
            if (step + 1) % config.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # Clipping de gradiente
                if config.fp16:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.max_grad_norm)
                
                # Atualizar parâmetros
                if config.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            # Atualizar barra de progresso
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": total_loss / (step + 1)})
            
            # Liberar memória
            torch.cuda.empty_cache()
        
        # Estatísticas da época
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_dataloader)
        
        print(f"Época {epoch+1} concluída em {epoch_time:.2f}s. Perda média: {avg_loss:.4f}")
    
    return student_model
```

## Conclusão

Nesta aula, exploramos técnicas essenciais para treinar LLMs de forma eficiente em hardware com limitações de memória. Aprendemos a:

1. Configurar hiperparâmetros adequados para GPUs com 4-6GB
2. Implementar estratégias de otimização como acumulação de gradientes e precisão mista
3. Monitorar e depurar o treinamento para identificar problemas
4. Aplicar técnicas avançadas como curriculum learning e distilação de conhecimento

Com essas técnicas, é possível treinar modelos surpreendentemente poderosos mesmo com recursos computacionais limitados. No próximo módulo, exploraremos técnicas de fine-tuning e adaptação para especializar nosso modelo em tarefas específicas.

## Exercícios Práticos

1. Configure um ambiente de treinamento com as otimizações discutidas e treine um modelo pequeno em um dataset de texto em português.
2. Experimente com diferentes configurações de hiperparâmetros e compare o impacto no uso de memória e velocidade de treinamento.
3. Implemente o monitoramento de gradientes e pesos durante o treinamento e analise os resultados.
4. Compare o desempenho do treinamento com e sem precisão mista (FP16).
5. Implemente curriculum learning em um dataset com textos de diferentes complexidades e observe o impacto na convergência do modelo.
