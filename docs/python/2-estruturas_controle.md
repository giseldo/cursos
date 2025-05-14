# Módulo 2: Estruturas de Controle

## Condicionais (if, else, elif)

As estruturas condicionais permitem que o programa tome decisões baseadas em condições.

### if simples
```python
idade = 18

if idade >= 18:
    print("Você é maior de idade")
```

### if-else
```python
idade = 16

if idade >= 18:
    print("Você é maior de idade")
else:
    print("Você é menor de idade")
```

### if-elif-else
```python
nota = 7.5

if nota >= 9:
    print("Conceito A")
elif nota >= 7:
    print("Conceito B")
elif nota >= 5:
    print("Conceito C")
else:
    print("Conceito D")
```

## Loops

### Loop for
O loop for é usado para iterar sobre uma sequência (lista, tupla, string, etc.).

```python
# Iterando sobre uma lista
frutas = ["maçã", "banana", "laranja"]
for fruta in frutas:
    print(fruta)

# Usando range()
for i in range(5):  # 0 a 4
    print(i)
```

### Loop while
O loop while executa um bloco de código enquanto uma condição for verdadeira.

```python
contador = 0
while contador < 5:
    print(contador)
    contador += 1
```

## Exercícios

1. Crie um programa que verifique se um número é par ou ímpar.

2. Faça um programa que imprima os números de 1 a 10, mas pule o número 5.

3. Crie um programa que calcule a soma dos números de 1 a 100 usando um loop.

4. Faça um programa que simule um jogo de adivinhação, onde o computador escolhe um número entre 1 e 100, e o usuário tenta adivinhar.

<!-- ## Soluções dos Exercícios

### Exercício 1
```python
numero = 7
if numero % 2 == 0:
    print(f"{numero} é par")
else:
    print(f"{numero} é ímpar")
```

### Exercício 2
```python
for i in range(1, 11):
    if i == 5:
        continue
    print(i)
```

### Exercício 3
```python
soma = 0
for i in range(1, 101):
    soma += i
print(f"A soma dos números de 1 a 100 é: {soma}")
```

### Exercício 4
```python
import random

numero_secreto = random.randint(1, 100)
tentativas = 0

while True:
    palpite = int(input("Digite um número entre 1 e 100: "))
    tentativas += 1
    
    if palpite < numero_secreto:
        print("Tente um número maior!")
    elif palpite > numero_secreto:
        print("Tente um número menor!")
    else:
        print(f"Parabéns! Você acertou em {tentativas} tentativas!")
        break
```  -->