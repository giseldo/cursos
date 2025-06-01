# Fine-tuning e Adaptação

Neste módulo, vamos explorar técnicas para adaptar e especializar nossos modelos de linguagem para tarefas específicas, utilizando métodos que são eficientes em termos de memória e computação. Aprenderemos a realizar fine-tuning completo e a utilizar técnicas avançadas como LoRA (Low-Rank Adaptation) e adaptadores, que são particularmente úteis para hardware com limitações de memória.

## Fine-tuning para Tarefas Específicas

O fine-tuning é o processo de continuar o treinamento de um modelo pré-treinado em um dataset específico para uma tarefa particular. Vamos explorar como realizar fine-tuning para diferentes tipos de tarefas.

### Adaptação para Classificação de Texto

Vamos começar com a adaptação de um LLM para tarefas de classificação:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class ClassificationHead(nn.Module):
    """
    Cabeça de classificação para adaptar um LLM para tarefas de classificação.
    """
    def __init__(self, hidden_size, num_classes, dropout_prob=0.1):
        """
        Inicializa a cabeça de classificação.
        
        Args:
            hidden_size: Dimensão dos estados ocultos do modelo
            num_classes: Número de classes para classificação
            dropout_prob: Probabilidade de dropout
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, hidden_states):
        """
        Forward pass da cabeça de classificação.
        
        Args:
            hidden_states: Estados ocultos do modelo de shape (batch_size, seq_len, hidden_size)
            
        Returns:
            torch.Tensor: Logits de classificação de shape (batch_size, num_classes)
        """
        # Usar apenas o último token para classificação
        pooled_output = hidden_states[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class LLMForClassification(nn.Module):
    """
    Adapta um LLM para tarefas de classificação.
    """
    def __init__(self, base_model, num_classes, freeze_base=True):
        """
        Inicializa o modelo para classificação.
        
        Args:
            base_model: Modelo base (LLM pré-treinado)
            num_classes: Número de classes para classificação
            freeze_base: Se True, congela os parâmetros do modelo base
        """
        super().__init__()
        self.base_model = base_model
        self.classification_head = ClassificationHead(
            hidden_size=base_model.config.hidden_size,
            num_classes=num_classes
        )
        
        # Congelar parâmetros do modelo base se solicitado
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass do modelo de classificação.
        
        Args:
            input_ids: Tensor de índices de tokens
            attention_mask: Máscara de atenção opcional
            labels: Labels de classificação opcionais
            
        Returns:
            dict: Dicionário contendo logits e perda (se labels fornecidos)
        """
        # Forward pass do modelo base
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Obter estados ocultos da última camada
        hidden_states = outputs.hidden_states[-1]
        
        # Classificação
        logits = self.classification_head(hidden_states)
        
        # Calcular perda se labels fornecidos
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states
        }

def fine_tune_for_classification(
    base_model_name,
    train_dataset,
    num_classes,
    output_dir,
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-5,
    freeze_base=True,
    device=None
):
    """
    Realiza fine-tuning de um LLM para classificação.
    
    Args:
        base_model_name: Nome ou caminho do modelo base
        train_dataset: Dataset de treinamento
        num_classes: Número de classes para classificação
        output_dir: Diretório para salvar o modelo
        num_epochs: Número de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        freeze_base: Se True, congela os parâmetros do modelo base
        device: Dispositivo para treinamento
        
    Returns:
        LLMForClassification: Modelo adaptado para classificação
    """
    # Configurar device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar modelo base e tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # Criar modelo para classificação
    model = LLMForClassification(
        base_model=base_model,
        num_classes=num_classes,
        freeze_base=freeze_base
    )
    model.to(device)
    
    # Configurar dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Configurar otimizador
    # Se o modelo base estiver congelado, otimizamos apenas a cabeça de classificação
    if freeze_base:
        optimizer = torch.optim.AdamW(model.classification_head.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Loop de treinamento
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Época {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Mover batch para device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Época {epoch+1}/{num_epochs}, Perda média: {avg_loss:.4f}")
    
    # Salvar modelo
    import os
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    return model
```

### Fine-tuning para Geração de Texto

Agora, vamos adaptar um LLM para tarefas de geração de texto:

```python
def fine_tune_for_generation(
    base_model_name,
    train_dataset,
    output_dir,
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    fp16=True,
    device=None
):
    """
    Realiza fine-tuning de um LLM para geração de texto.
    
    Args:
        base_model_name: Nome ou caminho do modelo base
        train_dataset: Dataset de treinamento
        output_dir: Diretório para salvar o modelo
        num_epochs: Número de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        gradient_accumulation_steps: Número de passos para acumulação de gradientes
        max_grad_norm: Norma máxima para clipping de gradiente
        fp16: Se True, usa precisão mista (FP16)
        device: Dispositivo para treinamento
        
    Returns:
        AutoModelForCausalLM: Modelo adaptado para geração
    """
    from torch.cuda.amp import autocast, GradScaler
    
    # Configurar device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar modelo base e tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Garantir que o tokenizer tenha token de padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Mover modelo para device
    model.to(device)
    
    # Habilitar gradient checkpointing para economizar memória
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Configurar dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Configurar otimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Configurar scaler para precisão mista
    scaler = GradScaler() if fp16 else None
    
    # Loop de treinamento
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Época {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Mover batch para device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Labels são os próprios input_ids para modelos de linguagem causal
            labels = input_ids.clone()
            
            # Forward pass com precisão mista se configurado
            if fp16:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass com scaler
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Atualizar parâmetros a cada gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # Clipping de gradiente
                if fp16:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Atualizar parâmetros
                if fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Época {epoch+1}/{num_epochs}, Perda média: {avg_loss:.4f}")
        
        # Salvar checkpoint
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
    
    # Salvar modelo final
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model
```

### Parameter-Efficient Fine-Tuning (PEFT)

O PEFT permite adaptar modelos grandes com muito menos parâmetros treináveis:

```python
def explain_peft():
    """
    Explica as técnicas de Parameter-Efficient Fine-Tuning (PEFT).
    """
    explanation = """
    # Parameter-Efficient Fine-Tuning (PEFT)
    
    PEFT refere-se a um conjunto de técnicas que permitem adaptar modelos grandes 
    treinando apenas uma pequena fração de seus parâmetros, economizando memória e 
    computação significativamente.
    
    ## Principais Técnicas PEFT:
    
    ### 1. Adaptadores
    
    - **Conceito**: Pequenas camadas inseridas entre as camadas existentes do modelo
    - **Funcionamento**: Projetam representações para um espaço de menor dimensão, aplicam transformação, e projetam de volta
    - **Vantagens**: Poucos parâmetros treináveis (tipicamente <1% do modelo)
    
    ```
    Original: Input → Layer → Output
    Com Adaptador: Input → Layer → Adapter → Output
    ```
    
    ### 2. LoRA (Low-Rank Adaptation)
    
    - **Conceito**: Aproxima as atualizações de matrizes de peso usando decomposição de baixo posto
    - **Funcionamento**: Para uma matriz W, treina matrizes A e B de baixo posto tal que W + AB substitui W
    - **Vantagens**: Muito eficiente em memória, pode ser aplicado seletivamente a camadas específicas
    
    ```
    Original: Y = WX
    Com LoRA: Y = WX + (AB)X, onde rank(AB) << min(dim(W))
    ```
    
    ### 3. Prefix Tuning / P-Tuning
    
    - **Conceito**: Adiciona vetores de prefixo treináveis às representações de cada camada
    - **Funcionamento**: Prepend tokens virtuais que são otimizados durante o fine-tuning
    - **Vantagens**: Preserva conhecimento do modelo original, eficiente para tarefas de geração
    
    ### 4. Prompt Tuning
    
    - **Conceito**: Treina embeddings de prompt contínuos (soft prompts)
    - **Funcionamento**: Adiciona tokens virtuais treináveis apenas na camada de entrada
    - **Vantagens**: Extremamente eficiente em parâmetros, fácil de implementar
    
    ## Comparação de Eficiência:
    
    | Técnica        | Parâmetros Treináveis | Overhead de Memória | Desempenho Relativo |
    |----------------|------------------------|---------------------|---------------------|
    | Fine-tuning    | 100%                   | Alto                | Excelente           |
    | Adaptadores    | ~0.5-3%                | Baixo               | Muito bom           |
    | LoRA           | ~0.1-1%                | Muito baixo         | Muito bom           |
    | Prefix Tuning  | ~0.1-0.5%              | Muito baixo         | Bom                 |
    | Prompt Tuning  | <0.1%                  | Mínimo              | Razoável            |
    
    ## Quando Usar Cada Técnica:
    
    - **LoRA**: Quando a memória é muito limitada mas o desempenho é crítico
    - **Adaptadores**: Para múltiplas tarefas com compartilhamento de conhecimento
    - **Prefix/Prompt Tuning**: Para cenários com muitas tarefas diferentes e troca rápida entre elas
    """
    
    return explanation
```

## LoRA e Adaptadores

Vamos implementar e explorar em detalhes as técnicas de LoRA e Adaptadores, que são particularmente úteis para hardware com limitações de memória.

### Implementação de LoRA (Low-Rank Adaptation)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """
    Implementação de uma camada LoRA (Low-Rank Adaptation).
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        """
        Inicializa a camada LoRA.
        
        Args:
            in_features: Número de features de entrada
            out_features: Número de features de saída
            rank: Posto da decomposição (r << min(in_features, out_features))
            alpha: Fator de escala para inicialização
            dropout: Taxa de dropout
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Matrizes de baixo posto
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Inicialização
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        """
        Forward pass da camada LoRA.
        
        Args:
            x: Tensor de entrada de shape (..., in_features)
            
        Returns:
            torch.Tensor: Resultado da adaptação LoRA
        """
        # Aplicar dropout na entrada
        x = self.dropout(x)
        
        # Multiplicação de baixo posto: x @ A @ B
        result = (x @ self.lora_A) @ self.lora_B
        
        # Aplicar escala
        return result * self.scaling

class LoRALinear(nn.Module):
    """
    Camada Linear com adaptação LoRA.
    """
    def __init__(self, linear_layer, rank=8, alpha=16, dropout=0.0, enable_lora=True):
        """
        Inicializa a camada Linear com LoRA.
        
        Args:
            linear_layer: Camada linear original
            rank: Posto da decomposição
            alpha: Fator de escala
            dropout: Taxa de dropout
            enable_lora: Se True, habilita LoRA
        """
        super().__init__()
        self.linear = linear_layer
        self.enable_lora = enable_lora
        
        if enable_lora:
            self.lora = LoRALayer(
                linear_layer.in_features,
                linear_layer.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
        
        # Congelar parâmetros da camada original
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass da camada Linear com LoRA.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            torch.Tensor: Resultado da camada
        """
        # Camada linear original
        result = self.linear(x)
        
        # Adicionar contribuição LoRA se habilitado
        if self.enable_lora:
            result = result + self.lora(x)
        
        return result

def apply_lora_to_model(model, target_modules=None, rank=8, alpha=16, dropout=0.0):
    """
    Aplica LoRA a um modelo.
    
    Args:
        model: Modelo a ser modificado
        target_modules: Lista de nomes de módulos para aplicar LoRA (None para todos os lineares)
        rank: Posto da decomposição
        alpha: Fator de escala
        dropout: Taxa de dropout
        
    Returns:
        nn.Module: Modelo com LoRA aplicado
    """
    if target_modules is None:
        # Por padrão, aplicar a todas as camadas lineares
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                target_modules.append(name)
    
    # Contador de parâmetros
    original_params = 0
    trainable_params = 0
    
    # Aplicar LoRA aos módulos alvo
    for name, module in model.named_modules():
        if name in target_modules and isinstance(module, nn.Linear):
            # Obter o módulo pai
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            # Substituir o módulo original pelo módulo LoRA
            original_params += module.in_features * module.out_features
            trainable_params += 2 * module.in_features * rank
            
            setattr(parent, child_name, LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            ))
    
    print(f"Parâmetros originais: {original_params}")
    print(f"Parâmetros treináveis com LoRA: {trainable_params}")
    print(f"Redução: {original_params / trainable_params:.2f}x")
    
    return model

def fine_tune_with_lora(
    base_model_name,
    train_dataset,
    output_dir,
    rank=8,
    alpha=16,
    target_modules=None,
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-4,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    fp16=True,
    device=None
):
    """
    Realiza fine-tuning de um LLM usando LoRA.
    
    Args:
        base_model_name: Nome ou caminho do modelo base
        train_dataset: Dataset de treinamento
        output_dir: Diretório para salvar o modelo
        rank: Posto da decomposição LoRA
        alpha: Fator de escala LoRA
        target_modules: Lista de nomes de módulos para aplicar LoRA
        num_epochs: Número de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        gradient_accumulation_steps: Número de passos para acumulação de gradientes
        max_grad_norm: Norma máxima para clipping de gradiente
        fp16: Se True, usa precisão mista (FP16)
        device: Dispositivo para treinamento
        
    Returns:
        tuple: (modelo base, adaptadores LoRA)
    """
    from torch.cuda.amp import autocast, GradScaler
    
    # Configurar device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar modelo base e tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Garantir que o tokenizer tenha token de padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Definir módulos alvo para LoRA se não especificados
    if target_modules is None:
        # Para modelos baseados em GPT, geralmente aplicamos LoRA às camadas de atenção
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                target_modules.append(name)
    
    # Aplicar LoRA ao modelo
    model = apply_lora_to_model(
        model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha
    )
    
    # Mover modelo para device
    model.to(device)
    
    # Configurar dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Configurar otimizador (apenas para parâmetros LoRA)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Configurar scaler para precisão mista
    scaler = GradScaler() if fp16 else None
    
    # Loop de treinamento
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Época {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Mover batch para device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Labels são os próprios input_ids para modelos de linguagem causal
            labels = input_ids.clone()
            
            # Forward pass com precisão mista se configurado
            if fp16:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass com scaler
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Atualizar parâmetros a cada gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # Clipping de gradiente
                if fp16:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in model.named_parameters() if p.requires_grad],
                    max_grad_norm
                )
                
                # Atualizar parâmetros
                if fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Época {epoch+1}/{num_epochs}, Perda média: {avg_loss:.4f}")
    
    # Salvar adaptadores LoRA
    os.makedirs(output_dir, exist_ok=True)
    
    # Extrair e salvar apenas os parâmetros LoRA
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            lora_state_dict[name] = param.data.cpu()
    
    torch.save(lora_state_dict, os.path.join(output_dir, "lora_adapters.pt"))
    
    # Salvar configuração LoRA
    lora_config = {
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_modules
    }
    
    import json
    with open(os.path.join(output_dir, "lora_config.json"), "w") as f:
        json.dump(lora_config, f, indent=2)
    
    # Salvar tokenizer
    tokenizer.save_pretrained(output_dir)
    
    return model, lora_state_dict

def load_model_with_lora(base_model_name, lora_dir):
    """
    Carrega um modelo base com adaptadores LoRA.
    
    Args:
        base_model_name: Nome ou caminho do modelo base
        lora_dir: Diretório contendo adaptadores LoRA
        
    Returns:
        nn.Module: Modelo com adaptadores LoRA aplicados
    """
    # Carregar modelo base
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # Carregar configuração LoRA
    import json
    with open(os.path.join(lora_dir, "lora_config.json"), "r") as f:
        lora_config = json.load(f)
    
    # Aplicar LoRA ao modelo
    model = apply_lora_to_model(
        model,
        target_modules=lora_config["target_modules"],
        rank=lora_config["rank"],
        alpha=lora_config["alpha"]
    )
    
    # Carregar parâmetros LoRA
    lora_state_dict = torch.load(os.path.join(lora_dir, "lora_adapters.pt"))
    
    # Carregar parâmetros no modelo
    missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)
    
    if len(missing) > 0:
        print(f"Parâmetros faltando: {len(missing)}")
    if len(unexpected) > 0:
        print(f"Parâmetros inesperados: {len(unexpected)}")
    
    return model
```

### Criação de Adaptadores

```python
class AdapterLayer(nn.Module):
    """
    Implementação de uma camada de Adaptador.
    """
    def __init__(self, hidden_size, adapter_size, dropout=0.1):
        """
        Inicializa a camada de Adaptador.
        
        Args:
            hidden_size: Dimensão do modelo
            adapter_size: Dimensão do gargalo do adaptador
            dropout: Taxa de dropout
        """
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Inicialização
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, hidden_states):
        """
        Forward pass da camada de Adaptador.
        
        Args:
            hidden_states: Estados ocultos do modelo
            
        Returns:
            torch.Tensor: Estados ocultos adaptados
        """
        # Salvar entrada para conexão residual
        residual = hidden_states
        
        # Projeção para baixo
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.activation(hidden_states)
        
        # Projeção para cima
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Conexão residual
        hidden_states = residual + hidden_states
        
        return hidden_states

class TransformerLayerWithAdapter(nn.Module):
    """
    Camada do Transformer com Adaptador.
    """
    def __init__(self, transformer_layer, adapter_size, dropout=0.1):
        """
        Inicializa a camada do Transformer com Adaptador.
        
        Args:
            transformer_layer: Camada original do Transformer
            adapter_size: Dimensão do gargalo do adaptador
            dropout: Taxa de dropout
        """
        super().__init__()
        self.layer = transformer_layer
        
        # Determinar hidden_size com base na camada
        # Isso depende da estrutura específica do modelo
        if hasattr(transformer_layer, "hidden_size"):
            hidden_size = transformer_layer.hidden_size
        elif hasattr(transformer_layer, "attention") and hasattr(transformer_layer.attention, "hidden_size"):
            hidden_size = transformer_layer.attention.hidden_size
        else:
            # Tentar inferir de alguma camada linear
            for module in transformer_layer.modules():
                if isinstance(module, nn.Linear):
                    hidden_size = module.out_features
                    break
            else:
                raise ValueError("Não foi possível determinar hidden_size")
        
        # Criar adaptadores
        self.attention_adapter = AdapterLayer(hidden_size, adapter_size, dropout)
        self.output_adapter = AdapterLayer(hidden_size, adapter_size, dropout)
        
        # Salvar método forward original
        self.original_forward = transformer_layer.forward
        
        # Congelar parâmetros da camada original
        for param in self.layer.parameters():
            param.requires_grad = False
    
    def forward(self, *args, **kwargs):
        """
        Forward pass da camada com adaptadores.
        
        Args:
            *args, **kwargs: Argumentos para a camada original
            
        Returns:
            Resultado da camada com adaptadores
        """
        # Chamar forward original
        outputs = self.original_forward(*args, **kwargs)
        
        # Extrair hidden states
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            rest = outputs[1:]
        else:
            hidden_states = outputs
            rest = tuple()
        
        # Aplicar adaptadores
        hidden_states = self.attention_adapter(hidden_states)
        hidden_states = self.output_adapter(hidden_states)
        
        # Reconstruir saída
        if isinstance(outputs, tuple):
            outputs = (hidden_states,) + rest
        else:
            outputs = hidden_states
        
        return outputs

def apply_adapters_to_model(model, adapter_size=64, dropout=0.1, target_modules=None):
    """
    Aplica adaptadores a um modelo.
    
    Args:
        model: Modelo a ser modificado
        adapter_size: Dimensão do gargalo do adaptador
        dropout: Taxa de dropout
        target_modules: Lista de nomes de módulos para aplicar adaptadores (None para todos)
        
    Returns:
        nn.Module: Modelo com adaptadores aplicados
    """
    if target_modules is None:
        # Por padrão, aplicar a todas as camadas do Transformer
        target_modules = []
        for name, module in model.named_modules():
            # Isso depende da estrutura específica do modelo
            if "layer" in name and any(x in name for x in ["attention", "output"]):
                target_modules.append(name)
    
    # Contador de parâmetros
    original_params = sum(p.numel() for p in model.parameters())
    
    # Aplicar adaptadores aos módulos alvo
    for name, module in model.named_modules():
        if name in target_modules:
            # Obter o módulo pai
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            # Substituir o módulo original pelo módulo com adaptador
            setattr(parent, child_name, TransformerLayerWithAdapter(
                module,
                adapter_size=adapter_size,
                dropout=dropout
            ))
    
    # Contar parâmetros treináveis
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Parâmetros totais: {original_params}")
    print(f"Parâmetros treináveis com adaptadores: {trainable_params}")
    print(f"Porcentagem de parâmetros treináveis: {trainable_params / original_params * 100:.2f}%")
    
    return model

def fine_tune_with_adapters(
    base_model_name,
    train_dataset,
    output_dir,
    adapter_size=64,
    dropout=0.1,
    target_modules=None,
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-4,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    fp16=True,
    device=None
):
    """
    Realiza fine-tuning de um LLM usando adaptadores.
    
    Args:
        base_model_name: Nome ou caminho do modelo base
        train_dataset: Dataset de treinamento
        output_dir: Diretório para salvar o modelo
        adapter_size: Dimensão do gargalo do adaptador
        dropout: Taxa de dropout
        target_modules: Lista de nomes de módulos para aplicar adaptadores
        num_epochs: Número de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        gradient_accumulation_steps: Número de passos para acumulação de gradientes
        max_grad_norm: Norma máxima para clipping de gradiente
        fp16: Se True, usa precisão mista (FP16)
        device: Dispositivo para treinamento
        
    Returns:
        tuple: (modelo base, adaptadores)
    """
    from torch.cuda.amp import autocast, GradScaler
    
    # Configurar device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar modelo base e tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Garantir que o tokenizer tenha token de padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Aplicar adaptadores ao modelo
    model = apply_adapters_to_model(
        model,
        adapter_size=adapter_size,
        dropout=dropout,
        target_modules=target_modules
    )
    
    # Mover modelo para device
    model.to(device)
    
    # Configurar dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Configurar otimizador (apenas para parâmetros dos adaptadores)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Configurar scaler para precisão mista
    scaler = GradScaler() if fp16 else None
    
    # Loop de treinamento
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Época {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Mover batch para device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Labels são os próprios input_ids para modelos de linguagem causal
            labels = input_ids.clone()
            
            # Forward pass com precisão mista se configurado
            if fp16:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass com scaler
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Atualizar parâmetros a cada gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # Clipping de gradiente
                if fp16:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in model.named_parameters() if p.requires_grad],
                    max_grad_norm
                )
                
                # Atualizar parâmetros
                if fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Época {epoch+1}/{num_epochs}, Perda média: {avg_loss:.4f}")
    
    # Salvar adaptadores
    os.makedirs(output_dir, exist_ok=True)
    
    # Extrair e salvar apenas os parâmetros dos adaptadores
    adapter_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            adapter_state_dict[name] = param.data.cpu()
    
    torch.save(adapter_state_dict, os.path.join(output_dir, "adapters.pt"))
    
    # Salvar configuração dos adaptadores
    adapter_config = {
        "adapter_size": adapter_size,
        "dropout": dropout,
        "target_modules": target_modules
    }
    
    import json
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
    
    # Salvar tokenizer
    tokenizer.save_pretrained(output_dir)
    
    return model, adapter_state_dict
```

## QLoRA para Hardware Limitado

QLoRA (Quantized LoRA) é uma técnica que combina quantização e LoRA para permitir fine-tuning em hardware extremamente limitado:

```python
def explain_qlora():
    """
    Explica a técnica QLoRA (Quantized LoRA).
    """
    explanation = """
    # QLoRA: Quantized Low-Rank Adaptation
    
    QLoRA é uma técnica que combina quantização de 4 bits com adaptação de baixo posto (LoRA),
    permitindo fine-tuning de modelos muito grandes em GPUs com memória limitada.
    
    ## Componentes Principais:
    
    ### 1. Quantização de 4 bits
    
    - **Conceito**: Reduz a precisão dos pesos do modelo de 16/32 bits para 4 bits
    - **Implementação**: Usa quantização NormalFloat-4 (NF4) otimizada para distribuições de pesos de LLMs
    - **Benefício**: Redução de 4-8x no uso de memória para armazenar o modelo base
    
    ### 2. Decomposição de Baixo Posto (LoRA)
    
    - **Conceito**: Aproxima as atualizações de matrizes de peso usando decomposição de baixo posto
    - **Implementação**: Para uma matriz W, treina matrizes A e B de baixo posto em precisão completa
    - **Benefício**: Redução significativa no número de parâmetros treináveis
    
    ### 3. Offloading para CPU
    
    - **Conceito**: Mantém partes do modelo na RAM da CPU, movendo para GPU apenas quando necessário
    - **Implementação**: Usa paginação para gerenciar transferências entre CPU e GPU
    - **Benefício**: Permite trabalhar com modelos maiores que a memória da GPU
    
    ### 4. Double Quantization
    
    - **Conceito**: Quantiza os fatores de quantização para economizar ainda mais memória
    - **Implementação**: Aplica uma segunda quantização aos fatores de escala da primeira quantização
    - **Benefício**: Economia adicional de memória sem perda significativa de qualidade
    
    ## Vantagens do QLoRA:
    
    1. **Eficiência de Memória**: Permite fine-tuning de modelos de 65B+ parâmetros em uma única GPU de 24GB
    2. **Preservação de Qualidade**: Mantém desempenho comparável ao fine-tuning completo
    3. **Velocidade Razoável**: Embora mais lento que LoRA padrão, ainda é prático para fine-tuning
    4. **Compatibilidade**: Os adaptadores treinados podem ser mesclados de volta ao modelo original
    
    ## Implementação com bibliotecas:
    
    ```python
    # Usando a biblioteca PEFT e bitsandbytes
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    
    # Configuração de quantização
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Carregar modelo quantizado
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Configuração LoRA
    lora_config = LoraConfig(
        r=16,                    # Dimensão do posto
        lora_alpha=32,           # Fator de escala
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Aplicar LoRA ao modelo quantizado
    model = get_peft_model(model, lora_config)
    ```
    
    ## Requisitos de Hardware:
    
    | Tamanho do Modelo | GPU Mínima Recomendada | Memória GPU |
    |-------------------|------------------------|-------------|
    | 7B                | RTX 3060 (6GB)         | 6GB         |
    | 13B               | RTX 3090 (24GB)        | 10GB        |
    | 33B               | RTX 4090 (24GB)        | 20GB        |
    | 65B               | A100 (40GB)            | 35GB        |
    
    ## Limitações:
    
    1. Treinamento mais lento devido às transferências CPU-GPU
    2. Requer bibliotecas específicas (bitsandbytes, PEFT)
    3. Algumas operações ainda precisam ser em precisão completa
    """
    
    return explanation

def setup_qlora_with_bitsandbytes():
    """
    Configura um ambiente para QLoRA usando bitsandbytes.
    """
    setup_code = """
    # Instalar bibliotecas necessárias
    !pip install -q bitsandbytes>=0.39.0
    !pip install -q transformers>=4.30.0
    !pip install -q peft>=0.4.0
    !pip install -q accelerate>=0.20.0
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    
    # Verificar se CUDA está disponível
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível. QLoRA requer GPU.")
    
    # Configuração de quantização de 4 bits
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Carregar modelo quantizado
    model_name = "meta-llama/Llama-2-7b-hf"  # Substitua pelo modelo desejado
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Carregar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Preparar modelo para treinamento em 4 bits
    model = prepare_model_for_kbit_training(model)
    
    # Configuração LoRA
    lora_config = LoraConfig(
        r=16,                    # Dimensão do posto
        lora_alpha=32,           # Fator de escala
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Aplicar LoRA ao modelo quantizado
    model = get_peft_model(model, lora_config)
    
    # Imprimir informações sobre parâmetros treináveis
    model.print_trainable_parameters()
    """
    
    return setup_code

def qlora_training_example():
    """
    Exemplo de código para treinamento com QLoRA.
    """
    training_code = """
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import load_dataset
    
    # Carregar dataset (exemplo com dataset pequeno)
    dataset = load_dataset("Abirate/english_quotes")
    
    # Função de pré-processamento
    def preprocess_function(examples):
        return tokenizer(
            examples["quote"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    # Processar dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Configurar data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Não usar mascaramento para modelos causais
    )
    
    # Configurar argumentos de treinamento
    training_args = TrainingArguments(
        output_dir="./qlora_output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        save_steps=500,
        logging_steps=100,
        fp16=True,  # Usar precisão mista
        optim="paged_adamw_8bit",  # Otimizador otimizado para memória
        report_to="none"
    )
    
    # Configurar trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Treinar modelo
    trainer.train()
    
    # Salvar adaptadores LoRA
    model.save_pretrained("./qlora_output/final_model")
    tokenizer.save_pretrained("./qlora_output/final_model")
    """
    
    return training_code
```

## Técnicas de Avaliação

É importante avaliar adequadamente os modelos após o fine-tuning:

```python
def evaluate_language_model(model, eval_dataset, tokenizer, device=None, batch_size=4, max_length=512):
    """
    Avalia um modelo de linguagem em termos de perplexidade.
    
    Args:
        model: Modelo a ser avaliado
        eval_dataset: Dataset de avaliação
        tokenizer: Tokenizer do modelo
        device: Dispositivo para avaliação
        batch_size: Tamanho do batch
        max_length: Comprimento máximo das sequências
        
    Returns:
        float: Perplexidade do modelo
    """
    # Configurar device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mover modelo para device
    model.to(device)
    model.eval()
    
    # Configurar dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Calcular perplexidade
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Avaliando"):
            # Mover batch para device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            # Calcular perda
            loss = outputs.loss
            
            # Contar tokens não mascarados
            num_tokens = attention_mask.sum().item() if attention_mask is not None else input_ids.numel()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calcular perplexidade
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def evaluate_classification_model(model, eval_dataset, device=None, batch_size=8):
    """
    Avalia um modelo de classificação.
    
    Args:
        model: Modelo a ser avaliado
        eval_dataset: Dataset de avaliação
        device: Dispositivo para avaliação
        batch_size: Tamanho do batch
        
    Returns:
        dict: Métricas de avaliação
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Configurar device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mover modelo para device
    model.to(device)
    model.eval()
    
    # Configurar dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Coletar predições
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Avaliando"):
            # Mover batch para device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Obter predições
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def analyze_model_bias_and_robustness(model, tokenizer, test_cases, device=None):
    """
    Analisa viés e robustez de um modelo.
    
    Args:
        model: Modelo a ser analisado
        tokenizer: Tokenizer do modelo
        test_cases: Lista de casos de teste
        device: Dispositivo para análise
        
    Returns:
        dict: Resultados da análise
    """
    # Configurar device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mover modelo para device
    model.to(device)
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        for category, cases in test_cases.items():
            category_results = []
            
            for case in cases:
                # Tokenizar entrada
                inputs = tokenizer(case["input"], return_tensors="pt").to(device)
                
                # Gerar saída
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                
                # Decodificar saída
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Analisar resultado
                analysis = {
                    "input": case["input"],
                    "output": generated_text,
                    "expected": case.get("expected"),
                    "pass": True
                }
                
                # Verificar se há palavras proibidas
                if "forbidden_words" in case:
                    for word in case["forbidden_words"]:
                        if word.lower() in generated_text.lower():
                            analysis["pass"] = False
                            analysis["reason"] = f"Contém palavra proibida: {word}"
                            break
                
                # Verificar se contém palavras obrigatórias
                if "required_words" in case:
                    for word in case["required_words"]:
                        if word.lower() not in generated_text.lower():
                            analysis["pass"] = False
                            analysis["reason"] = f"Não contém palavra obrigatória: {word}"
                            break
                
                category_results.append(analysis)
            
            # Calcular taxa de aprovação para a categoria
            pass_rate = sum(1 for r in category_results if r["pass"]) / len(category_results)
            
            results[category] = {
                "pass_rate": pass_rate,
                "details": category_results
            }
    
    return results
```

## Conclusão

Neste módulo, exploramos técnicas para adaptar e especializar LLMs para tarefas específicas, com foco em métodos eficientes para hardware com limitações de memória. Aprendemos a:

1. Realizar fine-tuning completo para tarefas de classificação e geração de texto
2. Implementar técnicas de Parameter-Efficient Fine-Tuning (PEFT)
3. Utilizar LoRA (Low-Rank Adaptation) para treinar modelos com poucos parâmetros
4. Criar e aplicar adaptadores para especialização de modelos
5. Utilizar QLoRA para hardware extremamente limitado
6. Avaliar modelos em termos de desempenho, viés e robustez

Estas técnicas permitem adaptar modelos de linguagem para uma ampla variedade de tarefas, mesmo com recursos computacionais limitados. No próximo módulo, exploraremos técnicas de inferência e otimização para utilizar os modelos treinados de forma eficiente.

## Exercícios Práticos

1. Implemente fine-tuning de um modelo pequeno (GPT-2 small ou similar) para classificação de sentimentos em português.
2. Compare o desempenho e uso de memória entre fine-tuning completo e LoRA em um modelo de tamanho médio.
3. Implemente QLoRA usando bitsandbytes para fine-tuning de um modelo maior em uma GPU com 6GB de memória.
4. Crie um conjunto de casos de teste para avaliar viés e robustez em um modelo adaptado.
5. Experimente com diferentes configurações de LoRA (rank, alpha, target_modules) e analise o impacto no desempenho e uso de memória.
