# Módulo 3: Funções e Módulos

## Funções

Funções são blocos de código reutilizáveis que realizam uma tarefa específica.

### Função Simples
```python
def saudacao():
    print("Olá, bem-vindo!")

# Chamando a função
saudacao()
```

### Função com Parâmetros
```python
def saudacao_personalizada(nome):
    print(f"Olá, {nome}! Bem-vindo!")

# Chamando a função
saudacao_personalizada("Maria")
```

### Função com Retorno
```python
def soma(a, b):
    return a + b

# Usando o retorno
resultado = soma(5, 3)
print(f"A soma é: {resultado}")
```

### Função com Múltiplos Parâmetros
```python
def calculadora(a, b, operacao='soma'):
    if operacao == 'soma':
        return a + b
    elif operacao == 'subtracao':
        return a - b
    elif operacao == 'multiplicacao':
        return a * b
    elif operacao == 'divisao':
        return a / b if b != 0 else "Erro: divisão por zero"
```

## Módulos

Módulos são arquivos Python que contêm funções, classes e variáveis que podem ser importados e usados em outros programas.

### Criando um Módulo
Crie um arquivo chamado `matematica.py`:
```python
def soma(a, b):
    return a + b

def subtracao(a, b):
    return a - b

def multiplicacao(a, b):
    return a * b

def divisao(a, b):
    return a / b if b != 0 else "Erro: divisão por zero"
```

### Importando um Módulo
```python
# Importando o módulo inteiro
import matematica

# Usando funções do módulo
resultado = matematica.soma(5, 3)

# Importando funções específicas
from matematica import soma, subtracao

# Usando as funções importadas
resultado = soma(5, 3)
```

## Exercícios

1. Crie uma função que calcule o fatorial de um número.

2. Faça uma função que verifique se um número é primo.

3. Crie um módulo chamado `geometria.py` com funções para calcular a área de diferentes formas geométricas (círculo, retângulo, triângulo).

4. Faça um programa que use o módulo `geometria.py` para calcular e exibir as áreas de diferentes formas.

## Soluções dos Exercícios

### Exercício 1
```python
def fatorial(n):
    if n == 0 or n == 1:
        return 1
    return n * fatorial(n - 1)

# Testando
print(f"Fatorial de 5: {fatorial(5)}")
```

### Exercício 2
```python
def eh_primo(numero):
    if numero < 2:
        return False
    for i in range(2, int(numero ** 0.5) + 1):
        if numero % i == 0:
            return False
    return True

# Testando
print(f"7 é primo? {eh_primo(7)}")
print(f"10 é primo? {eh_primo(10)}")
```

### Exercício 3 (geometria.py)
```python
import math

def area_circulo(raio):
    return math.pi * raio ** 2

def area_retangulo(base, altura):
    return base * altura

def area_triangulo(base, altura):
    return (base * altura) / 2
```

### Exercício 4
```python
from geometria import area_circulo, area_retangulo, area_triangulo

# Calculando áreas
print(f"Área do círculo (raio 5): {area_circulo(5):.2f}")
print(f"Área do retângulo (5x3): {area_retangulo(5, 3)}")
print(f"Área do triângulo (base 4, altura 6): {area_triangulo(4, 6)}") 