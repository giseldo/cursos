# Estrutura Detalhada do Curso: Construindo um LLM do Zero com Python

## Público-Alvo
- Desenvolvedores Python com conhecimento intermediário
- Estudantes e profissionais de IA e Aprendizado de Máquina
- Entusiastas de tecnologia com recursos computacionais limitados (GPU 4-6GB ou Google Colab)

## Pré-requisitos
- Conhecimento básico/intermediário de Python
- Noções fundamentais de álgebra linear e cálculo
- Familiaridade com conceitos básicos de aprendizado de máquina
- Computador com GPU de 4-6GB ou acesso ao Google Colab

## Estrutura do Curso

### Módulo 1: Introdução aos Modelos de Linguagem de Grande Escala
1. **O que são LLMs e como funcionam**
   - História e evolução dos modelos de linguagem
   - Arquiteturas principais: RNN, LSTM, Transformer
   - Diferenças entre GPT, BERT, T5 e outros modelos populares

2. **Requisitos de hardware e otimizações**
   - Entendendo as limitações de memória
   - Técnicas de otimização para hardware limitado
   - Configuração do ambiente no Google Colab vs. máquina local

3. **Fundamentos matemáticos e teóricos**
   - Revisão de álgebra linear relevante
   - Probabilidade e estatística para modelos de linguagem
   - Conceitos de atenção e auto-atenção

### Módulo 2: Preparação do Ambiente e Ferramentas
1. **Configuração do ambiente de desenvolvimento**
   - Instalação de dependências (PyTorch, Transformers, etc.)
   - Configuração de GPU e CUDA
   - Ferramentas de monitoramento de recursos

2. **Manipulação de dados textuais**
   - Coleta e preparação de datasets
   - Técnicas de limpeza e pré-processamento de texto
   - Criação de datasets de treinamento e validação

3. **Tokenização e vocabulário**
   - Implementação de tokenizadores simples
   - Byte-Pair Encoding (BPE) do zero
   - Integração com tokenizadores existentes

### Módulo 3: Arquitetura do Transformer
1. **Componentes fundamentais**
   - Embeddings de tokens e posicionais
   - Mecanismo de atenção multi-cabeça
   - Feed-forward networks e normalização

2. **Implementação do encoder**
   - Construção do bloco de atenção
   - Implementação da camada feed-forward
   - Montagem do encoder completo

3. **Implementação do decoder**
   - Atenção mascarada
   - Atenção cruzada encoder-decoder
   - Montagem do decoder completo

### Módulo 4: Construindo um LLM Compacto
1. **Arquitetura do modelo**
   - Definição da estrutura do modelo
   - Inicialização de parâmetros
   - Implementação do forward pass

2. **Técnicas de otimização de memória**
   - Quantização de pesos
   - Gradient checkpointing
   - Offloading para CPU

3. **Paralelismo e sharding**
   - Paralelismo de dados
   - Paralelismo de modelo
   - Técnicas de sharding para GPUs limitadas

### Módulo 5: Treinamento do Modelo
1. **Preparação para treinamento**
   - Definição de hiperparâmetros
   - Estratégias de otimização
   - Configuração de checkpoints

2. **Treinamento eficiente**
   - Implementação do loop de treinamento
   - Técnicas de acumulação de gradientes
   - Monitoramento e debugging

3. **Técnicas avançadas de treinamento**
   - Curriculum learning
   - Mixed-precision training
   - Distilação de conhecimento

### Módulo 6: Fine-tuning e Adaptação
1. **Fine-tuning para tarefas específicas**
   - Adaptação para classificação de texto
   - Fine-tuning para geração de texto
   - Parameter-Efficient Fine-Tuning (PEFT)

2. **LoRA e Adaptadores**
   - Implementação de LoRA (Low-Rank Adaptation)
   - Criação de adaptadores
   - QLoRA para hardware limitado

3. **Técnicas de avaliação**
   - Métricas de avaliação
   - Conjuntos de teste
   - Análise de viés e robustez

### Módulo 7: Inferência e Otimização
1. **Geração de texto eficiente**
   - Beam search e sampling
   - Técnicas de decoding
   - Controle de geração

2. **Otimização para inferência**
   - Pruning e quantização pós-treinamento
   - Exportação para ONNX
   - Inferência em CPU

3. **Deployment do modelo**
   - Empacotamento do modelo
   - Criação de API simples
   - Integração com aplicações

### Módulo 8: Projeto Final
1. **Definição do projeto**
   - Escolha do domínio e tarefa
   - Planejamento da implementação
   - Coleta de dados

2. **Implementação**
   - Construção do modelo
   - Treinamento e avaliação
   - Otimização e refinamento

3. **Apresentação e documentação**
   - Documentação do código
   - Análise de resultados
   - Demonstração do modelo

## Projetos Práticos por Módulo

1. **Módulo 1-2**: Implementação de um tokenizador BPE do zero
2. **Módulo 3**: Construção de um mini-transformer para classificação de sentimentos
3. **Módulo 4**: Implementação de um modelo de linguagem de pequena escala
4. **Módulo 5**: Treinamento de um modelo em um corpus específico
5. **Módulo 6**: Fine-tuning com LoRA para uma tarefa especializada
6. **Módulo 7**: Criação de uma API de inferência otimizada
7. **Módulo 8**: Projeto final completo - LLM especializado em um domínio

## Recursos Necessários

### Datasets
- Corpus de texto em português para treinamento inicial
- Datasets específicos para fine-tuning
- Conjuntos de teste para avaliação

### Código
- Implementações de referência para cada componente
- Notebooks para Google Colab
- Scripts de utilidade para processamento de dados

### Ferramentas
- Bibliotecas: PyTorch, Transformers, Accelerate, PEFT
- Ferramentas de monitoramento: Weights & Biases, TensorBoard
- Utilitários: ONNX Runtime, FlashAttention (versão leve)

## Cronograma Sugerido
- Módulos 1-2: 2 semanas
- Módulos 3-4: 3 semanas
- Módulos 5-6: 3 semanas
- Módulos 7-8: 4 semanas
