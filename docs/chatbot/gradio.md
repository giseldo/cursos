# Como criar um chatbot com o Gradio

::: info
Esta é uma tradução livre para o português brasileiro e com algumas alterações do excelente material construído pela equipe do [Gradio](https://www.gradio.app/guides/creating-a-chatbot-fast).
:::

## Introdução

Chatbots são aplicações populares de _Large Language Models_ (LLMs). Usando o Gradio, você pode facilmente construir uma demonstração de uso de um LLM e compartilhar com outros usuários, ou você pode conversar com o chatbot você mesmo utilizando uma intuitiva interface de conversação com o chatbot.

Este tutorial utiliza a classe `gr.ChatInterface()`, que é uma abstração de alto nível que permite a criação de uma interface gráfica rapidamente e com poucas linhas de código em python. O código construído pode ser facilmente adaptado para suportar chatbots multimodal (que suporta além de texto, também vídeo, audio e texto), ou chatbots que precisam de customização adicionais.

::: warning
Pré-requisitos: Verifique está utilizando a última versão do Gradio.
:::

Para instalar o Gradio, digite no seu terminal:

```shell
$ pip install --upgrade gradio
```

Para descobrir a versão instalada do Gradio usando o pip no Python, você pode usar o seguinte comando no terminal ou no ambiente onde está trabalhando:

### Comando no terminal

```shell
$ pip show gradio
```

Este comando exibirá informações sobre o pacote, incluindo a versão instalada.

### Saída esperada

Algo semelhante a isso será exibido:

```shell
Name: gradio
Version: X.Y.Z
Summary: A package for building machine learning demos and web apps
...
```

### Alternativa dentro do Python

Se preferir verificar diretamente no código Python, use o seguinte script:

```python
import gradio
print(gradio.__version__)
```

Isso imprimirá a versão instalada do Gradio.

## Desenhando uma função de chat

Quando trabalhamos com a classe `gr.ChatInterface()` a primeira coisa que precisamos definir é a **função chat**. Neste exemplo simples, sua função chat deve aceitar dois argumentos: **message** e **history** ( os argumentos podem ser nomeados com qualquer nome, mas precisam estar na ordem)

- **message**: uma **string** representando a mensagem mais recente do usuário
- **history**: uma lista no estilo do dicionário da openAI com as chaves **role** e **content**, representando o histórico da conversa. Pode também conter metadados das mensagens.

Por exemplo, **history** pode ser um dicionário deste tipo:

```shell
[
    {"role": "user", "content": "Qual é a capital do Brasil?"},
    {"role": "assistant", "content": "Rio de Janeiro"},
]
```

Sua função chat simplesmente precisa retornar:

- Um valor **string**, que é a respota do chatbot baseado no histórico (**history**) do chatbot e na mensagem mais recente.

## Um chatbot que responde randomicamente sim e não

Veja um exemplo de uma função chat, ela será chamada de `random_response`:

<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/giseldo/cursos/main?labpath=python/pln/cap1.ipynb) -->

```python
import random

def random_response(message, history):
    return random.choice(['sim', 'não'])
```

Agora podemos plugar esta função `random_response` no parâmetro `fn` no construtor da classe `gr.ChatInterface()` e chamar o método `.launch()` para criar um chatbot com uma interface web.

```python
import gradio as gr

gr.ChatInterface(
    fn=random_response,
    type="messages"
).launch()
```

A seguir o código completo:

```python
import random
import gradio as gr

def random_response(message, history):
    return random.choice(['sim', 'não'])

gr.ChatInterface(
    fn=random_response,
    type="messages"
).launch()
```

Para executar localmente, execute no terminal:

```shell
$ python app.py
```

Finalmente converse com este exemplo de chatbot em execução publicado no Hugging Face:

<iframe src="https://giseldo-chatinterface-random-response.hf.space" frameborder="0" width="100%" height="450"></iframe>

::: tip DICA
Sempre altere o valor do parametro `type="messages"` no `grChatInterface()`. O valor default (`type="tuples"`) está depreciado e será removido em versões futuras do Gradio.
:::

## Exemplo: um chatbot que alterna entre concordar e discordar

Este novo exemplo pega a entrada do usuário e retorna, além de utilizar o histórico.

```python
import gradio as gr

def alterna_concorda(message, history):
    if len([h for h in history if h['role'] == "assistant"]) % 2 == 0:
        return f"Sim, eu concordo com : {message}"
    else:
        return "Eu não concordo"

gr.ChatInterface(
    fn=random_response,
    type="messages"
).launch()
```

continua...
