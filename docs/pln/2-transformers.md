# Transformers

Os modelos transformers são usados para resolver todos os tipos de tarefas de PLN, como algumas mencionadas na seção anterior. Aqui estão algumas empresas e organizações usando a Hugging Face e os modelos Transformers. Estas empresas também contribuem de volta para a comunidade compartilhando seus modelos.

- Facebook AI - 23 modelos
- Microsoft - 33 modelos
- Grammarly - 1 modelo
- Google AI - 115 modelos
- Asteroid-team - 1 modelo
- Allen Institute AI - 43 modelos
- Typerform - modelos

A biblioteca [Transformers](https://github.com/huggingface/transformers) oferece a funcionalidade para criar e usar esses modelos compartilhados. O [Model Hub](https://huggingface.co/models) contém milhares de modelos pré-treinados que qualquer um pode baixar e usar. Você pode também dar upload nos seus próprios modelos no Hub!

O hugging Face Hub não é limitado aos modelos Transformers. Qualquer um pode compartilhar quaisquer tipos de modelos ou datasets que quiserem!

Exemplos:

O objeto mais básico na biblioteca Transformers é a função `pipeline()`. Ela conecta o modelo com seus passos necessários de pré e pós processamento, permitindo-nos diretamente inserir qualquer texto e obter uma resposta.

[Colab source-code](https://colab.research.google.com/drive/1BctAyiLiAerEyZxo3JwNqwcBXONv3D_t?usp=sharing)

```Python {filename="main.py"}
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've bee waiting for a HuggingFace course my whole life.")
```

```shell
[{'label': 'POSITIVE', 'score': 0.9516071081161499}]
```

O modelo cacheado default utilizado foi o _distilbert/distilbert-base-uncased-finetuned-sst-2-english_

Também podemos passar mais de uma sentença.

[Colab source-code](https://colab.research.google.com/drive/1BctAyiLiAerEyZxo3JwNqwcBXONv3D_t?usp=sharing)

```Python {filename="main.py"}
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(["I've bee waiting for a HuggingFace course my whole life.", "I hate this so much"])
```

```shell
[{'label': 'NEGATIVE', 'score': 0.6071575880050659},
 {'label': 'NEGATIVE', 'score': 0.9995144605636597}]
```

Por padrão, esse pipeline seleciona particularmente um modelo pré-treinado que tem sido _ajustado_ (fine-tuned) para análise de sentimentos em inglês. O modelo é baixado e cacheado quando você cria o objeto `classifier`. Se você rodar novamente o comando, o modelo cacheado será usado e não será baixado  novamente.

Há três principais passos envolvidos quando você passa algum texto para um pipeline:

1. O texto é pré-processado para um formato que o modelo consiga entender.
2. As entradas (_inputs_) pré-processados são passadas para o modelo.
3. As predições do modelo são pós-processadas, para que então você consiga atribuir sentido a elas.

Alguns dos pipelines disponíveis atualmente, são:

- _zero-shot-classification_ (classificação zero-shot)
- _text-generation_ (geração de texto)
- _feature-extraction_ (pega a representação vetorial do texto)
- _fill-mask_ (preenchimento de máscara)
- _ner_ (reconhecimento de entidades nomeadas)
- _question-answering_ (responder perguntas)
- _sentiment-analysis_ (análise de sentimentos)
- _summarization_ (sumarização)
- _translation_ (tradução)

## Classificação Zero-shot

Esse é um cenário comum nos projetos reais porque anotar texto geralmente consome bastante tempo e requer expertize no domínio. Para esse caso, o pipeline `zero-sho-classification` é muito poderoso: permite você especificar quais os rótulos usar para a classificação, desse modo você não precisa "confiar" nos rótulos pré-treinados. Você já viu como um modelo pode classificar uma sentença como positiva ou negativa usando esses dois rótulos - mas também pode ser classificado usando qualquer outro conjunto de rótulos que você quiser.

[Colab source-code](https://colab.research.google.com/drive/1BctAyiLiAerEyZxo3JwNqwcBXONv3D_t?usp=sharing)

```Python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier("This is a course about the Transformers library",
          candidate_labels=["education", "politics", "business"])
```

O modelo padrão utilizado foi _facebook/bart-large-mnli_

```shell
{'sequence': 'This is a course about the Transformers library',
 'labels': ['education', 'business', 'politics'],
 'scores': [0.8445994257926941, 0.11197380721569061, 0.04342673346400261]}
```

Esse pipeline é chamado de _zero-shot_ porque você não precisa fazer o ajuste fino do modelo nos dados que você o utiliza.

## Geração de Texto

A principal ideia aqui é que você coloque um pedaço de texto e o modelo irá autocompletá-lo ao gerar o texto restante.
Isso é similar ao recurso de predição textual que é encontrado em inúmeros celulares.
A geração de texto envolve aleatoriedade, então é normal se você não obter o mesmo resultado obtido mostrado abaixo.

[Colab source-code](https://colab.research.google.com/drive/1BctAyiLiAerEyZxo3JwNqwcBXONv3D_t?usp=sharing)

```Python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

O modelo padrão utilizado foi o _openai-community/gpt2_.

```shell
[{'generated_text': """In this course, we will teach you how to  navigate the real world using the virtual world which can serve  as a powerful tool to help you develop skills in the real world  and learn skills in the virtual world.\n\nFor the VirtualWorld course,"""}]
```

Você pode controlar quão diferentes sequências são geradas com o argumento `num_return_sequences` e o tamanho total da saída de texto (_output_) com o argumento `max_length`.

[Colab source-code](https://colab.research.google.com/drive/1BctAyiLiAerEyZxo3JwNqwcBXONv3D_t?usp=sharing)

```Python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to", num_return_sequences=2, max_length=30)
```

O modelo padrão utilizado foi o _openai-community/gpt2_.

```Python
[{'generated_text': """In this course, we will teach you how to develop and use your voice to help others around you understand how you can help them.\n\nThe"""},
 {'generated_text': """In this course, we will teach you how to  create multiple different web applications to run in multiple  languages, providing you a complete framework for writing an  application"""}]
 ```

### Usando qualquer modelo do Hub em um pipeline

Nos exemplos passados, usamos o modelo padrão para a tarefa que executamos, mas você pode usar um modelo particular do Hub para usá-lo no pipeline em uma tarefa específica, tal como, geração de texto (_text-generation_). Vá ao [Model Hub](https://huggingface.co/models) e clique _Edit filters_, selecione _text_generation_  na esquerda para mostrar apenas os modelos suportáveis para esta tarefa.

Vamos utilizar o modelo _distilgpt2_.

[Colab source-code](https://colab.research.google.com/drive/1BctAyiLiAerEyZxo3JwNqwcBXONv3D_t?usp=sharing)

```Python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator ("In this course, we will teach you how to",
          max_length=30,
          num_return_sequences=2
)
```

```shell
[{'generated_text': """In this course, we will teach you how to write a new language without using anything new. For example; as it is written, the same language"""},
 {'generated_text': """In this course, we will teach you how to  solve this problem through a real, real, and real data-centric approach: an algorithm that combines"""}]
```

Experimente! Use os filtros para encontrar um modelo de geração de texto em outra lingua no [Model Hub](https://huggingface.co/models). Veja na figura à seguir.

Uma vez que você seleciona o modelo clicando nele, você irá ver que há um widget que permite que você teste-o diretamente online. Desse modo você pode rapidamente testar as capacidades do modelo antes de baixa-lo. Veja na figura à seguir.
