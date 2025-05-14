# Módulo 5: Manipulação de Arquivos

## Leitura de Arquivos

### Lendo um arquivo de texto
```python
# Lendo todo o arquivo
with open('arquivo.txt', 'r', encoding='utf-8') as arquivo:
    conteudo = arquivo.read()
    print(conteudo)

# Lendo linha por linha
with open('arquivo.txt', 'r', encoding='utf-8') as arquivo:
    for linha in arquivo:
        print(linha.strip())  # strip() remove espaços em branco e quebras de linha

# Lendo todas as linhas em uma lista
with open('arquivo.txt', 'r', encoding='utf-8') as arquivo:
    linhas = arquivo.readlines()
```

## Escrita de Arquivos

### Escrevendo em um arquivo
```python
# Escrevendo texto
with open('novo_arquivo.txt', 'w', encoding='utf-8') as arquivo:
    arquivo.write('Primeira linha\n')
    arquivo.write('Segunda linha\n')

# Adicionando conteúdo (append)
with open('novo_arquivo.txt', 'a', encoding='utf-8') as arquivo:
    arquivo.write('Terceira linha\n')
```

## Manipulação de Arquivos CSV

### Trabalhando com arquivos CSV
```python
import csv

# Escrevendo em um arquivo CSV
with open('dados.csv', 'w', newline='', encoding='utf-8') as arquivo:
    escritor = csv.writer(arquivo)
    escritor.writerow(['Nome', 'Idade', 'Cidade'])
    escritor.writerow(['João', '25', 'São Paulo'])
    escritor.writerow(['Maria', '30', 'Rio de Janeiro'])

# Lendo um arquivo CSV
with open('dados.csv', 'r', encoding='utf-8') as arquivo:
    leitor = csv.reader(arquivo)
    for linha in leitor:
        print(linha)
```

## Manipulação de Arquivos JSON

### Trabalhando com arquivos JSON
```python
import json

# Escrevendo em um arquivo JSON
dados = {
    'nome': 'João',
    'idade': 25,
    'cidade': 'São Paulo',
    'hobbies': ['leitura', 'música', 'esportes']
}

with open('dados.json', 'w', encoding='utf-8') as arquivo:
    json.dump(dados, arquivo, indent=4)

# Lendo um arquivo JSON
with open('dados.json', 'r', encoding='utf-8') as arquivo:
    dados_lidos = json.load(arquivo)
    print(dados_lidos)
```

## Exercícios

1. Crie um programa que leia um arquivo de texto e conte a quantidade de palavras e linhas.

2. Faça um programa que crie um arquivo CSV com informações de produtos (nome, preço, quantidade) e depois leia e exiba essas informações.

3. Crie um programa que gerencie uma lista de tarefas (to-do list) usando um arquivo JSON.

4. Faça um programa que copie o conteúdo de um arquivo para outro, adicionando numeração nas linhas.

<!-- ## Soluções dos Exercícios

### Exercício 1
```python
def analisar_arquivo(nome_arquivo):
    with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
        conteudo = arquivo.read()
        linhas = conteudo.split('\n')
        palavras = conteudo.split()
        
        return {
            'linhas': len(linhas),
            'palavras': len(palavras)
        }

# Testando
resultado = analisar_arquivo('texto.txt')
print(f"Linhas: {resultado['linhas']}")
print(f"Palavras: {resultado['palavras']}")
```

### Exercício 2
```python
import csv

def criar_produtos():
    produtos = [
        ['Produto', 'Preço', 'Quantidade'],
        ['Notebook', '3500.00', '10'],
        ['Smartphone', '2000.00', '15'],
        ['Tablet', '1500.00', '8']
    ]
    
    with open('produtos.csv', 'w', newline='', encoding='utf-8') as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerows(produtos)

def ler_produtos():
    with open('produtos.csv', 'r', encoding='utf-8') as arquivo:
        leitor = csv.reader(arquivo)
        for linha in leitor:
            print(f"{linha[0]}: R${linha[1]} - {linha[2]} unidades")

# Testando
criar_produtos()
ler_produtos()
```

### Exercício 3
```python
import json

class GerenciadorTarefas:
    def __init__(self, arquivo):
        self.arquivo = arquivo
        self.tarefas = self.carregar_tarefas()
    
    def carregar_tarefas(self):
        try:
            with open(self.arquivo, 'r', encoding='utf-8') as arquivo:
                return json.load(arquivo)
        except FileNotFoundError:
            return []
    
    def salvar_tarefas(self):
        with open(self.arquivo, 'w', encoding='utf-8') as arquivo:
            json.dump(self.tarefas, arquivo, indent=4)
    
    def adicionar_tarefa(self, tarefa):
        self.tarefas.append({
            'tarefa': tarefa,
            'concluida': False
        })
        self.salvar_tarefas()
    
    def marcar_concluida(self, indice):
        if 0 <= indice < len(self.tarefas):
            self.tarefas[indice]['concluida'] = True
            self.salvar_tarefas()
    
    def listar_tarefas(self):
        for i, tarefa in enumerate(self.tarefas):
            status = '✓' if tarefa['concluida'] else ' '
            print(f"{i+1}. [{status}] {tarefa['tarefa']}")

# Testando
gerenciador = GerenciadorTarefas('tarefas.json')
gerenciador.adicionar_tarefa("Estudar Python")
gerenciador.adicionar_tarefa("Fazer exercícios")
gerenciador.listar_tarefas()
```

### Exercício 4
```python
def copiar_arquivo_numerado(origem, destino):
    with open(origem, 'r', encoding='utf-8') as arquivo_origem:
        with open(destino, 'w', encoding='utf-8') as arquivo_destino:
            for i, linha in enumerate(arquivo_origem, 1):
                arquivo_destino.write(f"{i}. {linha}")

# Testando
copiar_arquivo_numerado('texto.txt', 'texto_numerado.txt')
```  -->