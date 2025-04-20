# Usando chatGPT com LangChain

## Introdução

LangChain é uma biblioteca de software de código aberto projetada para
simplificar a interação com Large Language Models (LLMs) e construir
aplicativos de processamento de linguagem natural robustos. Ele fornece
uma camada de abstração de alto nível sobre as complexidades de
trabalhar diretamente com modelos de linguagem, tornando mais acessível
a criação de aplicativos de compreensão e geração de linguagem.

## Por que usar LangChain?

Trabalhar com LLMs pode ser complexo devido à sua natureza sofisticada e
aos requisitos de recursos computacionais. LangChain lida com muitos
detalhes complexos em segundo plano, permitindo que os desenvolvedores
se concentrem na construção de aplicativos de linguagem eficazes. Aqui
estão algumas vantagens do uso do LangChain:

- Simplicidade: LangChain oferece uma API simples e intuitiva, ocultando
  os detalhes complexos de interação com LLMs. Ele abstrai as nuances de
  carregar modelos, gerenciar recursos computacionais e executar
  previsões.

- Flexibilidade: A biblioteca suporta vários frameworks de deep
  learning, como TensorFlow e PyTorch, e pode ser integrada a diferentes
  LLMs. Isso oferece aos desenvolvedores a flexibilidade de escolher as
  ferramentas e modelos que melhor atendem às suas necessidades.

- Extensibilidade: LangChain é projetado para ser extensível, permitindo
  que os usuários criem seus próprios componentes personalizados. Você
  pode adicionar novos modelos, adaptar o processamento de texto ou
  desenvolver recursos específicos do domínio para atender aos
  requisitos exclusivos do seu aplicativo.

- Comunidade e suporte: LangChain tem uma comunidade ativa de
  desenvolvedores e pesquisadores que contribuem para o projeto. A
  documentação abrangente, tutoriais e suporte da comunidade tornam mais
  fácil começar e navegar por quaisquer desafios que surgirem durante o
  desenvolvimento.

## Arquitetura do LangChain

A arquitetura do LangChain pode ser entendida em três componentes
principais:

Camada de Abstração: Esta camada fornece uma interface simples e
unificada para interagir com diferentes LLMs. Ele abstrai as
complexidades de carregar, inicializar e executar previsões em modelos,
oferecendo uma API consistente independentemente do modelo subjacente.

Camada de Processamento de Texto: O LangChain inclui ferramentas
robustas para processamento de texto, incluindo tokenização, análise
sintática, reconhecimento de entidades nomeadas (NER) e muito mais. Esta
camada prepara os dados de entrada e saída para que possam ser
processados de forma eficaz pelos modelos de linguagem.

Camada de Modelo: Aqui é onde os próprios LLMs residem. O LangChain
suporta uma variedade de modelos de linguagem, desde modelos
pré-treinados de uso geral até modelos personalizados específicos de
domínio. Esta camada lida com a execução de previsões, gerenciamento de
recursos computacionais e interação com as APIs dos modelos.

## Exemplo Básico: Consultando um LLM

Vamos ver um exemplo simples de como usar o LangChain para consultar um
LLM e obter uma resposta. Neste exemplo, usaremos o gpt-4o-mini da
OpenAI, para responder a uma pergunta.

Primeiro, importe as bibliotecas necessárias e configure o cliente
LangChain. Em seguida, carregue o modelo de linguagem desejado. Agora,
você pode usar o modelo para fazer uma consulta. Vamos perguntar quem é
o presidente do Brasil.

[ ![image](./fig/colab-badge.png)
](https://colab.research.google.com/github/giseldo/chatbotbook_v2/blob/main/notebook/langchain.ipynb)

        from langchain.chat_models import init_chat_model
        import os

        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

        model = init_chat_model("gpt-4o-mini", model_provider="openai", 
            openai_api_key=OPENAI_API_KEY)

        response = model.invoke([
            {"role":"user", "content": "quem é o presidente do Brasil?"}
        ])

        print(response.content)

        print(response.text)

Este exemplo básico demonstra a simplicidade de usar o LangChain para
interagir com LLMs. No entanto, o LangChain oferece muito mais recursos
e funcionalidades para construir aplicativos de chatbot mais robustos.
