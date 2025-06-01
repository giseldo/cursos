# Manipulação de Dados Textuais

A qualidade e preparação dos dados são fundamentais para o sucesso de qualquer modelo de linguagem. Nesta aula, vamos explorar técnicas para coletar, limpar e pré-processar dados textuais, preparando-os para o treinamento de nosso LLM.

## Coleta e Preparação de Datasets

Antes de treinar um modelo de linguagem, precisamos de dados textuais de alta qualidade e em quantidade suficiente. Vamos explorar diferentes fontes e métodos para obtenção desses dados.

### Fontes de Dados Textuais

Existem diversas fontes de dados textuais que podem ser utilizadas para treinar LLMs:

1. **Datasets Públicos**:
   - **Hugging Face Datasets**: Biblioteca com centenas de datasets prontos para uso
   - **Common Crawl**: Corpus massivo de páginas web
   - **Wikipedia**: Conteúdo enciclopédico em múltiplos idiomas
   - **Project Gutenberg**: Livros de domínio público
   - **OpenWebText**: Recriação do dataset WebText usado para treinar o GPT-2

2. **Dados Específicos de Domínio**:
   - Artigos científicos (ex: arXiv, PubMed)
   - Documentação técnica
   - Código-fonte (para modelos que geram código)
   - Textos literários específicos

3. **Dados Multilíngues**:
   - OSCAR: Corpus web multilíngue
   - MC4: Multilingual C4, versão multilíngue do C4 (Colossal Clean Crawled Corpus)
   - OPUS: Coleção de textos traduzidos

Vamos ver como acessar alguns desses datasets usando a biblioteca Hugging Face Datasets:

```python
from datasets import load_dataset

# Carregar um subconjunto do Wikipedia em português
wiki_pt = load_dataset("wikipedia", "20220301.pt", split="train")
print(f"Número de artigos: {len(wiki_pt)}")
print(f"Colunas disponíveis: {wiki_pt.column_names}")
print(f"Exemplo de artigo: {wiki_pt[0]['title']}")

# Carregar um dataset de livros
books = load_dataset("bookcorpus", split="train")
print(f"Número de exemplos: {len(books)}")

# Carregar um dataset multilíngue
mc4_pt = load_dataset("mc4", "pt", split="train", streaming=True)
# Como este dataset é muito grande, usamos streaming
for i, example in enumerate(mc4_pt):
    if i < 3:  # Mostrar apenas os 3 primeiros exemplos
        print(f"Exemplo {i}: {example['text'][:100]}...")
    else:
        break
```

### Criação de Datasets Personalizados

Para casos específicos, podemos precisar criar nossos próprios datasets:

```python
import os
import glob
from datasets import Dataset

def create_dataset_from_text_files(directory_path, extension=".txt"):
    """
    Cria um dataset a partir de arquivos de texto em um diretório.
    
    Args:
        directory_path: Caminho para o diretório contendo os arquivos
        extension: Extensão dos arquivos a serem considerados
        
    Returns:
        Dataset: Um objeto Dataset do Hugging Face
    """
    file_paths = glob.glob(os.path.join(directory_path, f"*{extension}"))
    texts = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append({"text": text, "file_name": os.path.basename(file_path)})
        except Exception as e:
            print(f"Erro ao ler {file_path}: {e}")
    
    return Dataset.from_list(texts)

# Exemplo de uso
# custom_dataset = create_dataset_from_text_files("./data/raw/my_texts")
# print(f"Dataset criado com {len(custom_dataset)} documentos")
```

### Web Scraping para Coleta de Dados

Para coletar dados específicos da web, podemos usar técnicas de web scraping:

```python
import requests
from bs4 import BeautifulSoup
from datasets import Dataset

def scrape_articles(urls):
    """
    Extrai o texto principal de artigos a partir de uma lista de URLs.
    
    Args:
        urls: Lista de URLs para extrair conteúdo
        
    Returns:
        Dataset: Um objeto Dataset do Hugging Face
    """
    articles = []
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remover elementos que geralmente não contêm conteúdo principal
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extrair título
            title = soup.title.text.strip() if soup.title else "Sem título"
            
            # Extrair texto principal (isso varia muito dependendo do site)
            # Esta é uma abordagem simples que pode precisar ser adaptada
            paragraphs = soup.find_all('p')
            text = ' '.join([p.text.strip() for p in paragraphs])
            
            articles.append({
                "url": url,
                "title": title,
                "text": text
            })
            
        except Exception as e:
            print(f"Erro ao processar {url}: {e}")
    
    return Dataset.from_list(articles)

# Exemplo de uso
# urls = [
#     "https://pt.wikipedia.org/wiki/Inteligência_artificial",
#     "https://pt.wikipedia.org/wiki/Aprendizado_de_máquina"
# ]
# scraped_dataset = scrape_articles(urls)
# print(f"Artigos extraídos: {len(scraped_dataset)}")
```

## Técnicas de Limpeza e Pré-processamento de Texto

Dados brutos geralmente contêm ruído, formatação inconsistente e outros problemas que podem prejudicar o treinamento. Vamos explorar técnicas para limpar e pré-processar textos.

### Limpeza Básica de Texto

```python
import re
import unicodedata
import html

def clean_text(text):
    """
    Realiza limpeza básica em texto.
    
    Args:
        text: Texto a ser limpo
        
    Returns:
        str: Texto limpo
    """
    # Decodificar entidades HTML
    text = html.unescape(text)
    
    # Normalizar caracteres Unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remover URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Remover emails
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    # Normalizar quebras de linha
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Exemplo
# dirty_text = "Olá,   mundo!\nEste é um\r\nexemplo com <b>HTML</b> &amp; http://exemplo.com"
# clean = clean_text(dirty_text)
# print(f"Original: {dirty_text}")
# print(f"Limpo: {clean}")
```

### Filtragem de Conteúdo de Baixa Qualidade

É importante filtrar conteúdo que pode prejudicar o treinamento:

```python
def filter_low_quality_text(text, min_length=50, max_length=100000):
    """
    Filtra textos de baixa qualidade com base em heurísticas simples.
    
    Args:
        text: Texto a ser avaliado
        min_length: Comprimento mínimo aceitável
        max_length: Comprimento máximo aceitável
        
    Returns:
        bool: True se o texto passar nos filtros, False caso contrário
    """
    # Verificar comprimento
    if len(text) < min_length or len(text) > max_length:
        return False
    
    # Verificar proporção de caracteres alfanuméricos
    alpha_ratio = sum(c.isalnum() for c in text) / len(text) if len(text) > 0 else 0
    if alpha_ratio < 0.5:  # Menos de 50% de caracteres alfanuméricos
        return False
    
    # Verificar repetições excessivas
    if len(set(text)) / len(text) < 0.1:  # Menos de 10% de caracteres únicos
        return False
    
    # Verificar presença de frases completas (heurística simples)
    if text.count('.') < 1 and text.count('!') < 1 and text.count('?') < 1:
        return False
    
    return True

# Aplicar filtro a um dataset
def filter_dataset(dataset, text_column="text"):
    """
    Filtra um dataset removendo exemplos de baixa qualidade.
    
    Args:
        dataset: Dataset a ser filtrado
        text_column: Nome da coluna que contém o texto
        
    Returns:
        Dataset: Dataset filtrado
    """
    return dataset.filter(lambda example: filter_low_quality_text(example[text_column]))

# Exemplo
# filtered_dataset = filter_dataset(wiki_pt)
# print(f"Tamanho original: {len(wiki_pt)}")
# print(f"Tamanho após filtragem: {len(filtered_dataset)}")
```

### Normalização de Texto

A normalização ajuda a reduzir a variabilidade do texto:

```python
import unicodedata

def normalize_text(text, lowercase=True, remove_accents=False):
    """
    Normaliza texto aplicando transformações como lowercase e remoção de acentos.
    
    Args:
        text: Texto a ser normalizado
        lowercase: Se True, converte para minúsculas
        remove_accents: Se True, remove acentos
        
    Returns:
        str: Texto normalizado
    """
    if lowercase:
        text = text.lower()
    
    if remove_accents:
        # Decompor caracteres acentuados e remover os acentos
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
    
    return text

# Exemplo
# text = "Olá, Mundo! Este é um EXEMPLO de texto com ACENTUAÇÃO."
# normalized = normalize_text(text, lowercase=True, remove_accents=True)
# print(f"Original: {text}")
# print(f"Normalizado: {normalized}")
```

## Criação de Datasets de Treinamento e Validação

Após coletar e limpar os dados, precisamos organizá-los em conjuntos de treinamento e validação.

### Divisão de Datasets

```python
from datasets import Dataset
import numpy as np

def split_dataset(dataset, train_ratio=0.9, seed=42):
    """
    Divide um dataset em conjuntos de treinamento e validação.
    
    Args:
        dataset: Dataset a ser dividido
        train_ratio: Proporção do dataset para treinamento
        seed: Semente para reprodutibilidade
        
    Returns:
        tuple: (train_dataset, validation_dataset)
    """
    # Definir o gerador de números aleatórios para reprodutibilidade
    rng = np.random.default_rng(seed)
    
    # Gerar índices aleatórios
    indices = rng.permutation(len(dataset))
    train_size = int(len(dataset) * train_ratio)
    
    # Dividir índices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Criar datasets
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    
    return train_dataset, val_dataset

# Exemplo
# train_ds, val_ds = split_dataset(filtered_dataset)
# print(f"Tamanho do conjunto de treinamento: {len(train_ds)}")
# print(f"Tamanho do conjunto de validação: {len(val_ds)}")
```

### Formatação para Treinamento de LLM

Para treinar um LLM, precisamos formatar os dados de maneira específica:

```python
def format_for_causal_lm(dataset, text_column="text", max_length=1024):
    """
    Formata um dataset para treinamento de modelo de linguagem causal (como GPT).
    
    Args:
        dataset: Dataset a ser formatado
        text_column: Nome da coluna que contém o texto
        max_length: Comprimento máximo para truncamento
        
    Returns:
        Dataset: Dataset formatado
    """
    def _format(examples):
        texts = examples[text_column]
        # Adicionar token de fim de texto
        formatted_texts = [text[:max_length] + " <|endoftext|>" for text in texts]
        return {"formatted_text": formatted_texts}
    
    return dataset.map(_format, batched=True)

# Exemplo
# formatted_train_ds = format_for_causal_lm(train_ds)
# print(f"Exemplo formatado: {formatted_train_ds[0]['formatted_text'][:100]}...")
```

## Tokenização e Vocabulário

A tokenização é o processo de converter texto em sequências de tokens que o modelo pode processar. Vamos explorar diferentes abordagens de tokenização.

### Implementação de Tokenizadores Simples

Vamos começar com um tokenizador simples baseado em palavras:

```python
class SimpleWordTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        # Tokens especiais
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }
        
        # Inicializar vocabulário com tokens especiais
        for token, idx in self.special_tokens.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        
        self.vocab_size = len(self.special_tokens)
    
    def build_vocab(self, texts, min_freq=2, max_vocab_size=50000):
        """
        Constrói o vocabulário a partir de uma lista de textos.
        
        Args:
            texts: Lista de textos
            min_freq: Frequência mínima para incluir uma palavra
            max_vocab_size: Tamanho máximo do vocabulário
        """
        # Contar frequência das palavras
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Filtrar por frequência mínima
        word_counts = {word: count for word, count in word_counts.items() 
                      if count >= min_freq}
        
        # Ordenar por frequência (decrescente)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Limitar ao tamanho máximo do vocabulário
        sorted_words = sorted_words[:max_vocab_size - len(self.special_tokens)]
        
        # Adicionar ao vocabulário
        for word, _ in sorted_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text, add_special_tokens=True):
        """
        Converte texto em uma sequência de IDs de tokens.
        
        Args:
            text: Texto a ser tokenizado
            add_special_tokens: Se True, adiciona tokens especiais
            
        Returns:
            list: Lista de IDs de tokens
        """
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.special_tokens["<bos>"])
        
        for word in text.split():
            if word in self.word_to_id:
                tokens.append(self.word_to_id[word])
            else:
                tokens.append(self.special_tokens["<unk>"])
        
        if add_special_tokens:
            tokens.append(self.special_tokens["<eos>"])
        
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Converte uma sequência de IDs de tokens de volta para texto.
        
        Args:
            token_ids: Lista de IDs de tokens
            skip_special_tokens: Se True, ignora tokens especiais
            
        Returns:
            str: Texto decodificado
        """
        special_ids = list(self.special_tokens.values()) if skip_special_tokens else []
        words = [self.id_to_word[idx] for idx in token_ids 
                if idx in self.id_to_word and (not skip_special_tokens or idx not in special_ids)]
        
        return " ".join(words)

# Exemplo de uso
# tokenizer = SimpleWordTokenizer()
# sample_texts = ["Este é um exemplo de texto.", "Outro exemplo para construir o vocabulário."]
# tokenizer.build_vocab(sample_texts)
# print(f"Tamanho do vocabulário: {tokenizer.vocab_size}")
# encoded = tokenizer.encode("Este é um exemplo.")
# print(f"Texto codificado: {encoded}")
# decoded = tokenizer.decode(encoded)
# print(f"Texto decodificado: {decoded}")
```

### Byte-Pair Encoding (BPE) do Zero

O BPE é um algoritmo de tokenização mais avançado que aprende a dividir palavras em subpalavras:

```python
from collections import defaultdict

class SimpleBPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.merges = {}
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }
        
        # Inicializar vocabulário com tokens especiais
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
        
        self.vocab_size = len(self.special_tokens)
    
    def _get_stats(self, words):
        """
        Conta a frequência de pares de símbolos adjacentes.
        """
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def _merge_vocab(self, pair, words):
        """
        Mescla um par de símbolos no vocabulário.
        """
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in words.items():
            parts = word.split()
            i = 0
            new_parts = []
            
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == pair[0] and parts[i + 1] == pair[1]:
                    new_parts.append(replacement)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            
            new_word = ' '.join(new_parts)
            new_words[new_word] = freq
        
        return new_words
    
    def train(self, texts, vocab_size=1000, min_frequency=2):
        """
        Treina o tokenizador BPE.
        
        Args:
            texts: Lista de textos para treinamento
            vocab_size: Tamanho desejado do vocabulário
            min_frequency: Frequência mínima para considerar uma palavra
        """
        # Inicializar com caracteres individuais
        word_freqs = defaultdict(int)
        for text in texts:
            for word in text.split():
                # Representar cada palavra como sequência de caracteres
                char_word = ' '.join(list(word))
                word_freqs[char_word] += 1
        
        # Filtrar por frequência mínima
        word_freqs = {word: freq for word, freq in word_freqs.items() 
                     if freq >= min_frequency}
        
        # Adicionar caracteres únicos ao vocabulário
        chars = set()
        for word in word_freqs.keys():
            for char in word.split():
                chars.add(char)
        
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = self.vocab_size
                self.vocab_size += 1
        
        # Realizar mesclagens BPE
        num_merges = vocab_size - self.vocab_size
        for i in range(num_merges):
            if not word_freqs:
                break
                
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
                
            # Encontrar par mais frequente
            best_pair = max(pairs, key=pairs.get)
            
            # Mesclar o par no vocabulário
            word_freqs = self._merge_vocab(best_pair, word_freqs)
            
            # Registrar a mesclagem
            self.merges[best_pair] = i
            
            # Adicionar o novo token ao vocabulário
            new_token = ''.join(best_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = self.vocab_size
                self.vocab_size += 1
            
            if self.vocab_size >= vocab_size:
                break
    
    def _tokenize_word(self, word):
        """
        Tokeniza uma única palavra usando as regras BPE aprendidas.
        """
        if not word:
            return []
            
        # Inicializar com caracteres
        chars = list(word)
        
        # Aplicar mesclagens na ordem em que foram aprendidas
        while len(chars) > 1:
            pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
            
            # Encontrar o par com a maior prioridade (aprendido primeiro)
            best_pair = None
            best_rank = float('inf')
            
            for pair in pairs:
                if pair in self.merges and self.merges[pair] < best_rank:
                    best_pair = pair
                    best_rank = self.merges[pair]
            
            if best_pair is None:
                break
                
            # Aplicar a mesclagem
            i = 0
            new_chars = []
            
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == best_pair[0] and chars[i + 1] == best_pair[1]:
                    new_chars.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            
            chars = new_chars
        
        return chars
    
    def encode(self, text, add_special_tokens=True):
        """
        Tokeniza um texto usando BPE.
        
        Args:
            text: Texto a ser tokenizado
            add_special_tokens: Se True, adiciona tokens especiais
            
        Returns:
            list: Lista de IDs de tokens
        """
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.special_tokens["<bos>"])
        
        for word in text.split():
            word_tokens = self._tokenize_word(word)
            
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.special_tokens["<unk>"])
        
        if add_special_tokens:
            tokens.append(self.special_tokens["<eos>"])
        
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decodifica uma sequência de IDs de tokens.
        
        Args:
            token_ids: Lista de IDs de tokens
            skip_special_tokens: Se True, ignora tokens especiais
            
        Returns:
            str: Texto decodificado
        """
        special_ids = list(self.special_tokens.values()) if skip_special_tokens else []
        
        # Inverter o mapeamento de vocabulário
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        tokens = []
        for idx in token_ids:
            if idx in id_to_token and (not skip_special_tokens or idx not in special_ids):
                tokens.append(id_to_token[idx])
        
        # Juntar tokens (simplificado - na prática, precisaríamos de regras mais complexas)
        return ''.join(tokens).replace('</w>', ' ').strip()

# Exemplo de uso
# bpe_tokenizer = SimpleBPETokenizer()
# sample_texts = ["Este é um exemplo de texto para treinar o BPE.",
#                "Precisamos de várias frases para que o algoritmo aprenda padrões úteis."]
# bpe_tokenizer.train(sample_texts, vocab_size=100)
# print(f"Tamanho do vocabulário BPE: {bpe_tokenizer.vocab_size}")
# encoded = bpe_tokenizer.encode("Este exemplo")
# print(f"Texto codificado com BPE: {encoded}")
# decoded = bpe_tokenizer.decode(encoded)
# print(f"Texto decodificado: {decoded}")
```

### Integração com Tokenizadores Existentes

Na prática, geralmente usamos tokenizadores já implementados e otimizados:

```python
from transformers import AutoTokenizer

def setup_tokenizer(model_name="gpt2"):
    """
    Configura um tokenizador pré-treinado.
    
    Args:
        model_name: Nome do modelo cujo tokenizador será usado
        
    Returns:
        tokenizer: Tokenizador configurado
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Garantir que temos os tokens especiais necessários
    special_tokens = {
        'pad_token': '<pad>',
        'eos_token': '<|endoftext|>',
        'bos_token': '<|startoftext|>'
    }
    
    # Adicionar tokens especiais que estão faltando
    tokenizer.add_special_tokens({
        k: v for k, v in special_tokens.items() 
        if getattr(tokenizer, k) is None
    })
    
    return tokenizer

# Exemplo de uso
# tokenizer = setup_tokenizer("gpt2")
# sample_text = "Este é um exemplo de texto para tokenização."
# encoded = tokenizer(sample_text, return_tensors="pt")
# print(f"Input IDs: {encoded.input_ids}")
# decoded = tokenizer.decode(encoded.input_ids[0])
# print(f"Texto decodificado: {decoded}")
```

## Criação de Datasets de Treinamento e Validação

Finalmente, vamos preparar nossos dados para treinamento, aplicando tokenização e formatação adequada:

```python
from datasets import Dataset
import torch

def prepare_dataset_for_training(texts, tokenizer, max_length=512, batch_size=4):
    """
    Prepara um dataset para treinamento de LLM.
    
    Args:
        texts: Lista de textos
        tokenizer: Tokenizador a ser usado
        max_length: Comprimento máximo das sequências
        batch_size: Tamanho do batch
        
    Returns:
        dataset: Dataset preparado para treinamento
    """
    # Criar dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Função de tokenização
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Aplicar tokenização
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"]
    )
    
    # Formatar para treinamento de modelo causal
    def format_for_clm(examples):
        examples["labels"] = examples["input_ids"].clone()
        return examples
    
    lm_dataset = tokenized_dataset.map(
        format_for_clm,
        batched=True
    )
    
    return lm_dataset

# Exemplo de uso
# tokenizer = setup_tokenizer("gpt2")
# sample_texts = ["Este é o primeiro exemplo.", "Este é o segundo exemplo."]
# train_dataset = prepare_dataset_for_training(sample_texts, tokenizer)
# print(f"Exemplo de batch: {train_dataset[0]}")
```

## Conclusão

Nesta aula, exploramos técnicas essenciais para coletar, limpar e preparar dados textuais para o treinamento de LLMs. Aprendemos a:

1. Coletar dados de diversas fontes, incluindo datasets públicos e web scraping
2. Limpar e pré-processar textos para remover ruído e inconsistências
3. Implementar tokenizadores simples e BPE do zero
4. Integrar com tokenizadores existentes
5. Preparar datasets formatados para treinamento de modelos de linguagem

Na próxima aula, vamos explorar a implementação da arquitetura Transformer, que é o coração dos LLMs modernos.

## Exercícios Práticos

1. Colete um pequeno corpus de textos em português (por exemplo, artigos da Wikipedia) e aplique as técnicas de limpeza discutidas.
2. Implemente o tokenizador simples baseado em palavras e treine-o no corpus coletado.
3. Compare os resultados da tokenização usando seu tokenizador com um tokenizador pré-treinado como o do GPT-2.
4. Experimente diferentes parâmetros de filtragem e limpeza e observe o impacto na qualidade dos dados.
5. Prepare um pequeno dataset de treinamento e validação usando o tokenizador do GPT-2.
