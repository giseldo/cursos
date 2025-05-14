# Módulo 4: Estruturas de Dados

## Listas

Listas são coleções ordenadas e mutáveis de elementos.

### Criando e Manipulando Listas
```python
# Criando uma lista
frutas = ["maçã", "banana", "laranja"]

# Acessando elementos
primeira_fruta = frutas[0]  # "maçã"
ultima_fruta = frutas[-1]   # "laranja"

# Adicionando elementos
frutas.append("uva")
frutas.insert(1, "morango")

# Removendo elementos
frutas.remove("banana")
fruta_removida = frutas.pop()  # Remove e retorna o último elemento

# Ordenando
frutas.sort()  # Ordem alfabética
frutas.reverse()  # Inverte a ordem
```

## Tuplas

Tuplas são coleções ordenadas e imutáveis de elementos.

### Trabalhando com Tuplas
```python
# Criando uma tupla
coordenadas = (10, 20)

# Acessando elementos
x = coordenadas[0]  # 10
y = coordenadas[1]  # 20

# Tuplas são imutáveis
# coordenadas[0] = 15  # Isso causará um erro
```

## Dicionários

Dicionários são coleções de pares chave-valor.

### Usando Dicionários
```python
# Criando um dicionário
pessoa = {
    "nome": "João",
    "idade": 25,
    "cidade": "São Paulo"
}

# Acessando valores
nome = pessoa["nome"]
idade = pessoa.get("idade")

# Adicionando ou modificando valores
pessoa["profissao"] = "Programador"
pessoa["idade"] = 26

# Removendo itens
del pessoa["cidade"]
profissao = pessoa.pop("profissao")
```

## Sets

Sets são coleções não ordenadas de elementos únicos.

### Operações com Sets
```python
# Criando sets
numeros = {1, 2, 3, 4, 5}
pares = {2, 4, 6, 8}

# Operações de conjunto
uniao = numeros | pares  # {1, 2, 3, 4, 5, 6, 8}
intersecao = numeros & pares  # {2, 4}
diferenca = numeros - pares  # {1, 3, 5}

# Adicionando e removendo elementos
numeros.add(6)
numeros.remove(1)
```

## Exercícios

1. Crie uma lista de números e implemente funções para:
   - Encontrar o maior e o menor número
   - Calcular a média
   - Encontrar números pares e ímpares

2. Crie um dicionário para armazenar informações de alunos (nome, notas, média) e implemente funções para:
   - Adicionar um novo aluno
   - Calcular a média de um aluno
   - Encontrar o aluno com a maior média

3. Crie um programa que use sets para encontrar elementos únicos em duas listas.

4. Faça um programa que simule uma agenda de contatos usando um dicionário.

## Soluções dos Exercícios

### Exercício 1
```python
def analisar_numeros(numeros):
    maior = max(numeros)
    menor = min(numeros)
    media = sum(numeros) / len(numeros)
    pares = [n for n in numeros if n % 2 == 0]
    impares = [n for n in numeros if n % 2 != 0]
    
    return {
        "maior": maior,
        "menor": menor,
        "media": media,
        "pares": pares,
        "impares": impares
    }

# Testando
numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
resultado = analisar_numeros(numeros)
print(resultado)
```

### Exercício 2
```python
class GerenciadorAlunos:
    def __init__(self):
        self.alunos = {}
    
    def adicionar_aluno(self, nome, notas):
        media = sum(notas) / len(notas)
        self.alunos[nome] = {
            "notas": notas,
            "media": media
        }
    
    def calcular_media(self, nome):
        return self.alunos[nome]["media"]
    
    def aluno_maior_media(self):
        return max(self.alunos.items(), key=lambda x: x[1]["media"])[0]

# Testando
gerenciador = GerenciadorAlunos()
gerenciador.adicionar_aluno("João", [8, 7, 9])
gerenciador.adicionar_aluno("Maria", [9, 9, 10])
print(f"Aluno com maior média: {gerenciador.aluno_maior_media()}")
```

### Exercício 3
```python
def elementos_unicos(lista1, lista2):
    set1 = set(lista1)
    set2 = set(lista2)
    
    apenas_lista1 = set1 - set2
    apenas_lista2 = set2 - set1
    em_ambas = set1 & set2
    
    return {
        "apenas_lista1": apenas_lista1,
        "apenas_lista2": apenas_lista2,
        "em_ambas": em_ambas
    }

# Testando
lista1 = [1, 2, 3, 4, 5]
lista2 = [4, 5, 6, 7, 8]
print(elementos_unicos(lista1, lista2))
```

### Exercício 4
```python
class Agenda:
    def __init__(self):
        self.contatos = {}
    
    def adicionar_contato(self, nome, telefone, email):
        self.contatos[nome] = {
            "telefone": telefone,
            "email": email
        }
    
    def buscar_contato(self, nome):
        return self.contatos.get(nome, "Contato não encontrado")
    
    def remover_contato(self, nome):
        if nome in self.contatos:
            del self.contatos[nome]
            return True
        return False
    
    def listar_contatos(self):
        return self.contatos

# Testando
agenda = Agenda()
agenda.adicionar_contato("João", "123-456", "joao@email.com")
agenda.adicionar_contato("Maria", "789-012", "maria@email.com")
print(agenda.listar_contatos())
``` 