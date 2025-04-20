# Chatbot ELIZA em Python

## Introdução

Apresenta-se, nesta seção, uma implementação simplificada em Python de
um chatbot inspirado no paradigma ELIZA. Esta implementação demonstra a
utilização de expressões regulares para a identificação de padrões
textuais (palavras-chave) na entrada fornecida pelo usuário e a
subsequente geração de respostas, fundamentada em regras de
transformação predefinidas manualmente.

[ ![image](./fig/colab-badge.png)
](https://colab.research.google.com/github/giseldo/chatbotbook/blob/main/notebook/eliza.ipynb)

``` {.python language="Python" caption="Chatbot Eliza em Python"}
import re  
import random  

regras = [
    (re.compile(r'\b(hello|hi|hey)\b', re.IGNORECASE),
     ["Hello. How do you do. Please tell me your problem."]),

    (re.compile(r'\b(I am|I\'?m) (.+)', re.IGNORECASE),
     ["How long have you been {1}?",   
      "Why do you think you are {1}?"]),

    (re.compile(r'\bI need (.+)', re.IGNORECASE),
     ["Why do you need {1}?",
      "Would it really help you to get {1}?"]),

    (re.compile(r'\bI can\'?t (.+)', re.IGNORECASE),
     ["What makes you think you can't {1}?",
      "Have you tried {1}?"]),

    (re.compile(r'\bmy (mother|father|mom|dad)\b', re.IGNORECASE),
     ["Tell me more about your family.",
      "How do you feel about your parents?"]),

    (re.compile(r'\b(sorry)\b', re.IGNORECASE),
     ["Please don't apologize."]),

    (re.compile(r'\b(maybe|perhaps)\b', re.IGNORECASE),
     ["You don't seem certain."]),

    (re.compile(r'\bbecause\b', re.IGNORECASE),
     ["Is that the real reason?"]),

    (re.compile(r'\b(are you|do you) (.+)\?$', re.IGNORECASE),
     ["Why do you ask that?"]),

    (re.compile(r'\bcomputer\b', re.IGNORECASE),
     ["Do computers worry you?"]),
]

respostas_padrao = [
    "I see.",  
    "Please tell me more.",  
    "Can you elaborate on that?"  
]

def response(entrada_usuario):
    for padrao, respostas in regras:
        match = padrao.search(entrada_usuario)  
        if match:
            resposta = random.choice(respostas)
            if match.groups():
                resposta = resposta.format(*match.groups())
            return resposta
    return random.choice(respostas_padrao)
```

``` {.python language="Python" caption="Exemplo de uso do chatbot ELIZA"}
print("User: Hello.")
print("Bot: " + response("Hello."))

print("User: I am feeling sad.")
print("Bot: " + response("I am feeling sad."))

print("Maybe I was not good enough.")
print("Bot: " + response("Maybe I was not good enough."))

print("My mother tried to help.")
print("Bot: " + response("My mother tried to help."))
```

Na implementação, são definidos múltiplos padrões de expressões
regulares que correspondem a palavras-chave ou estruturas frasais de
interesse (e.g., saudações, construções como "I am" ou "I need",
referências a termos familiares). A função `response`, ao receber uma
string de entrada, itera sequencialmente sobre essas regras. Para cada
regra, utiliza-se o método `padrao.search(entrada)` para verificar a
ocorrência do padrão correspondente na sentença do usuário.

Quando uma correspondência (match) é encontrada, uma resposta associada
à regra é selecionada. É relevante notar que diversas respostas
predefinidas contêm marcadores (placeholders), como `{1}`. Estes
marcadores são dinamicamente substituídos pelo conteúdo dos grupos
capturados pela expressão regular na entrada original do usuário. Por
exemplo, se a entrada for *"I am sad"* e esta corresponder a um padrão
como `r"I am (.*)"`, o grupo capturado (`(.*)`) conterá a string
*"sad"*. Se a resposta associada for *"Why do you think you are {1}?"*,
o sistema a formatará como *"Why do you think you are sad?"*,
substituindo `{1}` pelo conteúdo do primeiro grupo capturado.

Na ausência de correspondência com qualquer regra específica, o sistema
seleciona aleatoriamente uma resposta genérica a partir de um conjunto
predefinido (denominado, por exemplo, `respostas_padrao`), tal como *"I
see."*.

## Exemplo de Interação

    Usuário: Hello
    Bot: Hello. How do you do. Please tell me your problem.

    Usuário: I am feeling sad
    Bot: Why do you think you are feeling sad?

    Usuário: Because I lost my job
    Bot: Is that the real reason?

    Usuário: Maybe I was not good enough.
    Bot: You don't seem certain.

    Usuário: My mother tried to help.
    Bot: Tell me more about your family.

Observa-se na interação que o chatbot inicialmente respondeu à saudação
(*"Hello\..."*) com base na regra 1. Em seguida, a entrada *"I am sad"*
ativou a regra 2, resultando em uma resposta que incorpora o termo
*"sad"*. A interação prossegue com o sistema respondendo a entradas
iniciadas por *"Because\..."* (regra 8) e *"Maybe\..."* (regra 7), entre
outras. Cada resposta gerada foi determinada pela regra correspondente
ativada e pelo conteúdo específico da sentença do usuário, mimetizando o
comportamento baseado em padrões do sistema ELIZA original
[@Abdul-Kader2015].
