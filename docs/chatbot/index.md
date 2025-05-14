---
lang: pt-BR
title: Tutorial de Chatbots em Python
viewport: width=device-width, initial-scale=1.0
---

# Tutorial de Chatbots em Python

Aprenda a Construir Chatbots Interativos com Python

## Introdução aos Chatbots

Um chatbot é uma aplicação de software projetada para simular uma
conversa humana. Este tutorial oferece um guia passo a passo para criar
chatbots utilizando Python, aproveitando a biblioteca Natural Language
Toolkit (NLTK) para processamento de linguagem natural. Ao final, você
terá construído um chatbot básico baseado em regras e um chatbot
avançado com capacidades de correspondência de padrões, além de
completar exercícios práticos.

## Configurando o Ambiente

Para começar, certifique-se de que o Python está instalado em seu
sistema (recomenda-se a versão 3.6 ou superior). Você também precisará
instalar a biblioteca NLTK, utilizada para tarefas de processamento de
linguagem natural.

### Etapas de Instalação

1.  Instale o Python a partir de
    [python.org](https://www.python.org/downloads/){target="_blank"
    rel="noopener noreferrer"}.

2.  Abra um terminal ou prompt de comando e instale o NLTK usando pip:
```bash
pip install nltk
```
3.  Inicie o Python em seu terminal e baixe os dados do NLTK:

```Python
import nltk
nltk.download('punkt')
```

## Construindo um Chatbot Básico

O primeiro chatbot utilizará uma lógica baseada em regras simples para
responder a entradas do usuário com base em padrões predefinidos. Abaixo
está um exemplo de um chatbot básico que responde a saudações e
perguntas comuns.

```Python
import random

# Define pares de respostas
responses = {
    "olá": ["Olá!", "Oi!", "E aí!"],
    "como você está": ["Estou ótimo, obrigado!", "Bem, e você?"],
    "tchau": ["Adeus!", "Até mais!"],
    "default": ["Desculpe, não entendi.", "Pode reformular isso?"]
}

def basic_chatbot(user_input):
    user_input = user_input.lower().strip()
    for key in responses:
        if key in user_input:
            return random.choice(responses[key])
    return random.choice(responses["default"])

# Loop principal
print("Chatbot Básico: Digite 'sair' para encerrar.")
while True:
    user_input = input("Você: ")
    if user_input.lower() == "sair":
        print("Chatbot: Adeus!")
        break
    print("Chatbot:", basic_chatbot(user_input))
```             

Este chatbot verifica se a entrada do usuário contém palavras-chave
específicas e responde com uma resposta aleatória da lista
correspondente. Se nenhuma palavra-chave for encontrada, ele fornece uma
resposta padrão.

## Construindo um Chatbot Avançado com NLTK

Para um chatbot mais sofisticado, utilizaremos a ferramenta de chat do
NLTK, que suporta correspondência de padrões para conversas mais
dinâmicas. O exemplo abaixo demonstra um chatbot que lida com uma
variedade de entradas do usuário usando padrões de expressões regulares.

```Python
from nltk.chat.util import Chat, reflections

# Define pares de conversa
pairs = [
    [r"meu nome é (.*)", ["Olá %1, prazer em conhecê-lo!", "Oi %1, como posso ajudá-lo?"]],
    [r"oi|olá|e aí", ["Olá!", "Oi!", "E aí!"]],
    [r"como você está", ["Estou bem, obrigado!", "Ótimo, e você?"]],
    [r"qual é o seu nome", ["Sou GrokBot, seu assistente virtual!", "Pode me chamar de GrokBot!"]],
    [r"sair|tchau|adeus", ["Adeus!", "Até mais!"]],
    [r"(.*)", ["Não sei se entendi.", "Pode explicar melhor?"]]
]

# Cria instância do chatbot
chatbot = Chat(pairs, reflections)

def advanced_chatbot():
    print("Chatbot Avançado: Digite 'sair' para encerrar.")
    chatbot.converse()

# Executa o chatbot
if __name__ == "__main__":
    advanced_chatbot()
```              

Este chatbot utiliza a classe \`Chat\` do NLTK para combinar entradas do
usuário com padrões de expressões regulares e responder adequadamente. O
dicionário \`reflections\` permite que o chatbot transforme declarações
(por exemplo, \"eu sou\" para \"você é\") para respostas mais naturais.

## Exercícios

Os exercícios a seguir ajudarão a reforçar seu entendimento sobre o
desenvolvimento de chatbots. Tente resolver cada exercício e verifique a
solução, se necessário.


### Exercício 1: Aprimorar o Chatbot Básico

Modifique o chatbot básico para incluir pelo menos três new pares de
respostas (por exemplo, para \"obrigado\", \"que horas são\" e
\"ajuda\"). Teste o chatbot para garantir que ele responde
apropriadamente.

Mostrar Solução

```Python
import random

responses = {
    "olá": ["Olá!", "Oi!", "E aí!"],
    "como você está": ["Estou ótimo, obrigado!", "Bem, e você?"],
    "tchau": ["Adeus!", "Até mais!"],
    "obrigado": ["De nada!", "Sem problemas!"],
    "que horas são": ["Não tenho relógio, mas é sempre hora de conversar!", "Hora de papear!"],
    "ajuda": ["Estou aqui para ajudar! Pergunte qualquer coisa.", "Como posso te ajudar?"],
    "default": ["Desculpe, não entendi.", "Pode reformular isso?"]
}

def basic_chatbot(user_input):
    user_input = user_input.lower().strip()
    for key in responses:
        if key in user_input:
            return random.choice(responses[key])
    return random.choice(responses["default"])

print("Chatbot Básico Aprimorado: Digite 'sair' para encerrar.")
while True:
    user_input = input("Você: ")
    if user_input.lower() == "sair":
        print("Chatbot: Adeus!")
        break
    print("Chatbot:", basic_chatbot(user_input))
```                        

### Exercício 2: Adicionar Padrões ao Chatbot Avançado

Amplie o chatbot avançado adicionando dois new pares de padrão-resposta.
Por exemplo, adicione um padrão para \"eu gosto de (.\*)\" e \"o que é
(.\*)\". Teste o chatbot para verificar as new respostas.

Mostrar Solução

```Python
from nltk.chat.util import Chat, reflections

pairs = [
    [r"meu nome é (.*)", ["Olá %1, prazer em conhecê-lo!", "Oi %1, como posso ajudá-lo?"]],
    [r"oi|olá|e aí", ["Olá!", "Oi!", "E aí!"]],
    [r"como você está", ["Estou bem, obrigado!", "Ótimo, e você?"]],
    [r"qual é o seu nome", ["Sou GrokBot, seu assistente virtual!", "Pode me chamar de GrokBot!"]],
    [r"sair|tchau|adeus", ["Adeus!", "Até mais!"]],
    [r"eu gosto de (.*)", ["Que legal! Eu também gosto de %1!", "Bacana, conte mais sobre %1!"]],
    [r"o que é (.*)", ["Não sou uma enciclopédia, mas %1 parece interessante!", "Pode contar mais sobre %1?"]],
    [r"(.*)", ["Não sei se entendi.", "Pode explicar melhor?"]]
]

chatbot = Chat(pairs, reflections)

def advanced_chatbot():
    print("Chatbot Avançado: Digite 'sair' para encerrar.")
    chatbot.converse()

if __name__ == "__main__":
    advanced_chatbot()
```                        
