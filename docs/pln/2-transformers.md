# Transformers

Os modelos treinados disponíveis na API `transformers` são usados para muitas tarefas de PLN. 

A biblioteca [transformers](https://github.com/huggingface/transformers) oferece a funcionalidade para criar e usar esses modelos compartilhados. O [Hugging Face Model Hub](https://huggingface.co/models) contém milhares de modelos pré-treinados que qualquer um pode baixar e usar. 

O [Hugging Face Model Hub](https://huggingface.co/models) não é limitado aos modelos `transformers`. Qualquer um pode compartilhar quaisquer tipos de modelos ou datasets que quiserem! Você mesmo pode fazer  upload nos seus próprios modelos no Hub!

:::info
Veja os modelos que Giseldo Neo já hospedou no Hugging Face, acessando o link http://huggingface.co/giseldo
:::

## Pipeline

O objeto mais básico na biblioteca `transformers` é a função `pipeline()`. Ela conecta o modelo com seus passos necessários de **pré** e **pós** processamento. Permitindo com facilidade inserir uma sentença ou texto e obter uma resposta da tarefa de PLN selecionada.

Há três principais passos envolvidos quando você passa algum texto para um pipeline:

1. O texto é pré-processado para um formato que o modelo consiga entender.
2. As entradas pré-processados são passadas para o modelo.
3. As predições do modelo são pós-processadas, para que então você consiga atribuir sentido a elas.

Alguns dos pipelines disponíveis atualmente, são:

- Análise de sentimentos `sentiment-analysis` 
- Classificação zero-shot `zero-shot-classification`
- Geração de texto `text-generation`
- Extrair representação vetorial do texto `feature-extraction`
- Sumarização `summarization`
- Tradução `translation`
- Responder perguntas `question-answering`
- Preenchimento de máscara `fill-mask`
- Reconhecimento de entidades nomeadas `ner`


Para alguns modelos, antes é necessário efetuar o login no Hugging Face. Você cria uma chave de api no site do Hugging Face e depois no terminal digital 

```shell
# Faça login no Hugging Face para acessar modelos privados 
# e informe a chave quando solicitado
huggingface-cli login
```

## Análise de sentimento

### Exemplo 1 - Análise de sentimento de uma sentença

Abaixo o código para usar um modelo classificador de sentimento. Basta informar um texto (uma sentença, um conjunto de frases) e passar para o pipeline que ele retorna o rótulo: `positivo` ou `negativo`, além do `score`, sendo o score confiança da rotulagem dos dados. O retorno é em formato JSON.

Antes, uma vez, instale as dependências necessárias no python. 

<ColabButton href="https://colab.research.google.com/github/giseldo/cursos/blob/main/docs/pln/notebook/transformer.ipynb" />

```shell
# dependências
pip install transformers
pip install tensorflow
pip install tf-keras
pip install torch
```

```Python
# classificador de sentimentos em textos em inglês
# default model distilbert/distilbert-base-uncased-finetuned-sst-2-english  
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've bee waiting for a HuggingFace course my whole life.")
```

```
[{'label': 'POSITIVE', 'score': 0.9516071081161499}]
```

Por padrão, se não for informado um modelo, o pipeline seleciona um modelo pré-treinado que foi _ajustado_ (fine-tuned) para análise de sentimentos em inglês, neste caso foi o `distilbert/distilbert-base-uncased-finetuned-sst-2-english`. 

O modelo é baixado e cacheado quando você cria o objeto `classifier`. Se você rodar novamente o comando, o modelo cacheado será usado e não será baixado novamente.

### Exemplo 2 - Análise de sentimento de uma lista de sentenças.

Também podemos passar mais de uma sentença em uma lista para o classificador, conforme exemplo a seguir.

<ColabButton href="https://colab.research.google.com/github/giseldo/cursos/blob/main/docs/pln/notebook/transformer.ipynb" />

```Python
# classificação de sentimento de uma lista de frases
# default distilbert/distilbert-base-uncased-finetuned-sst-2-english
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier([
    "I've bee waiting for a HuggingFace course my whole life.", 
    "I hate this so much"
])
```

```
[{'label': 'NEGATIVE', 'score': 0.6071575880050659},
 {'label': 'NEGATIVE', 'score': 0.9995144605636597}]
```

### Exemplo 3 - Análise de sentimento informando um modelo específico

Agora um exemplo em português, desta vez informando um modelo que suporta o idioma português brasileiro.

<ColabButton href="https://colab.research.google.com/github/giseldo/cursos/blob/main/docs/pln/notebook/transformer.ipynb" />

```Python
# exemplo em português brasileiro
from transformers import pipeline

classifier = pipeline("sentiment-analysis", 
  model="clapAI/modernBERT-large-multilingual-sentiment")

saida = classifier("Eu estou feliz.")
```

```
[{'label': 'positive', 'score': 0.8990556597709656}]
```

## Classificação Zero-shot

Esse é um cenário comum em alguns projetos porque anotar texto geralmente consome bastante tempo e requer expertize no domínio. Para esse caso, o pipeline `zero-sho-classification` é muito poderoso: permite você especificar quais os rótulos usar para a classificação, desse modo você não precisa "confiar" nos rótulos pré-treinados. Você já viu como um modelo pode classificar uma sentença como positiva ou negativa usando esses dois rótulos - mas também pode ser classificado usando qualquer outro conjunto de rótulos que você quiser.

<ColabButton href="https://colab.research.google.com/github/giseldo/cursos/blob/main/docs/pln/notebook/transformer.ipynb" />

```Python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier("This is a course about the Transformers library",
          candidate_labels=["education", "politics", "business"])
```

```
{'sequence': 'This is a course about the Transformers library',
 'labels': ['education', 'business', 'politics'],
 'scores': [0.8445994257926941, 0.11197380721569061, 0.04342673346400261]}
```

O modelo padrão para a tarefa `zero-shot-classification` quiando não informado o modelo foi o _facebook/bart-large-mnli_

Esse pipeline é chamado de _zero-shot_ porque você não precisa fazer o ajuste fino do modelo nos dados que você o utiliza.

## Geração de Texto

A principal ideia aqui é que você coloque um pedaço de texto e o modelo irá autocompletá-lo ao gerar o texto restante.
Isso é similar ao recurso de predição textual que é encontrado em inúmeros celulares.
A geração de texto envolve aleatoriedade, então é normal se você não obter o mesmo resultado obtido mostrado abaixo.

<ColabButton href="https://colab.research.google.com/github/giseldo/cursos/blob/main/docs/pln/notebook/transformer.ipynb" />

```Python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

```
 [{'generated_text': """In this course, we will teach 
you how to  navigate the real world using the virtual 
world which can serve  as a powerful tool to help you
develop skills in the real world  and learn skills in 
the virtual world.\n\nFor the VirtualWorld course,"""}]
``` 

O modelo padrão para a tarefa `text-generation` é o _openai-community/gpt2_.

Você pode controlar quão diferentes sequências são geradas com o argumento `num_return_sequences` e o tamanho total da saída de texto (_output_) com o argumento `max_length`.

<ColabButton href="https://colab.research.google.com/github/giseldo/cursos/blob/main/docs/pln/notebook/transformer.ipynb" />

```Python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to", 
    num_return_sequences=2, 
    max_length=30)
```

``` 
[{'generated_text': """In this course, we will teach 
you how to develop and use your voice to help others 
around you understand how you can help them.\n\nThe"""},
 {'generated_text': """In this course, we will teach 
 you how to  create multiple different web applications 
 to run in multiple  languages, providing you a complete 
 framework for writing an  application"""}]
``` 

O modelo padrão utilizado foi o _openai-community/gpt2_.    

:::info
Nos exemplos passados, usamos o modelo padrão para a tarefa de PLN que executamos, mas você pode usar um modelo particular do [Hugging Face Model Hub](https://huggingface.co/models) para usá-lo no pipeline em uma tarefa específica, tal como, geração de texto (_text-generation_). Vá ao 
[Hugging Face Model Hub](https://huggingface.co/models) e clique _Edit filters_, selecione _text_generation_  na esquerda para mostrar apenas os modelos suportáveis para esta tarefa.
:::

Vamos utilizar o modelo _distilgpt2_.

<ColabButton href="https://colab.research.google.com/github/giseldo/cursos/blob/main/docs/pln/notebook/transformer.ipynb" />

```Python
from transformers import pipeline

generator = pipeline("text-generation", 
  model="distilgpt2")
generator ("In this course, we will teach you how to",
          max_length=30,
          num_return_sequences=2)
```

```
[{'generated_text': '''In this course, we will teach you 
how to write a new language without using anything new. 
For example; as it is written, the same language'''},
 {'generated_text': '''In this course, we will teach you 
 how to  solve this problem through a real, real, and real 
 data-centric approach: an algorithm that combines'''}]
```

:::info
Experimente! Use os filtros para encontrar um modelo de geração de texto em outra língua no [Hugging Face Model Hub](https://huggingface.co/models). 
:::

Uma vez que você seleciona o modelo clikando nele, você irá ver que há um widget que permite que você teste-o diretamente online. Desse modo você pode rapidamente testar as capacidades do modelo antes de baixá-lo. Veja na figura a seguir.

## Preenchimento de máscara

O próximo pipeline que você irá testar é o fill-mask. A ideia dessa tarefa é preencher os espaços em branco com um texto dado:

<ColabButton href="https://colab.research.google.com/github/giseldo/cursos/blob/main/docs/pln/notebook/transformer.ipynb" />

```Python
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

```
[{'sequence': 'This course will teach you all about mathematical models.',
  'score': 0.19619831442832947,
  'token': 30412,
  'token_str': ' mathematical'},
 {'sequence': 'This course will teach you all about computational models.',
  'score': 0.04052725434303284,
  'token': 38163,
  'token_str': ' computational'}]
```

O argumento `top_k` controla quantas possibilidades você quer que sejam geradas. Note que aqui o modelo recebe uma entrada com uma palavra `<mask>` especial, que é frequentemente referida como `mask token`. Outros modelos de preenchimento de máscara podem ter diferentes `mask tokens`, então é sempre bom verificar quais são os `mask tokens` apropriados quando explorar outros modelos. Um modo de checar isso é olhando para os `mask tokens` usados nos widget do Hugging Face, disponível na página do modelo utilizado no Hugging Face.

:::info
Experimente! Pesquise pelo modelo `bert-base-cased` no [Hugging Face Model Hub](https://huggingface.co/models) e identifique suas palavras `mask tokens` no widget da API de inferência. O que esse modelo prediz para a sentença em nosso pipeline no exemplo acima?
:::

Quando for utilizado o modelo `bert-base-cased` vai dar erro pois a mask token é ` [MASK]` e não `<mask>`.

<ColabButton href="https://colab.research.google.com/github/giseldo/cursos/blob/main/docs/pln/notebook/transformer.ipynb" />

```Python
# exemplo de geração de texto com bert-base-cased
from transformers import pipeline

unmasker = pipeline("fill-mask", 
                    model="bert-base-cased")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

## Reconhecimento de entidades nomeadas

Reconhecimento de Entidades Nomeadas (NER) é uma tarefa onde o modelo tem de achar quais partes do texto correspondem a entidades como pessoas, locais, organizações. Vamos olhar em um exemplo:

<ColabButton href="https://colab.research.google.com/github/giseldo/cursos/blob/main/docs/pln/notebook/transformer.ipynb" />

```Python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

```
[{'entity_group': 'PER', 'score': 0.99816, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
 {'entity_group': 'ORG', 'score': 0.97960, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
 {'entity_group': 'LOC', 'score': 0.99321, 'word': 'Brooklyn', 'start': 49, 'end': 57}
]
```

Aqui o modelo corretamente identificou que Sylvain é uma pessoa (PER), Hugging Face é uma organização (ORG), e Brooklyn é um local (LOC).

Nós passamos a opção `grouped_entities=True` na criação da função do pipelina para dize-lo para reagrupar juntos as partes da sentença que correspondem à mesma entidade: aqui o modelo agrupou corretamente “Hugging” e “Face” como única organização, ainda que o mesmo nome consista em múltiplas palavras. Na verdade, como veremos no próximo capítulo, o pré-processamento até mesmo divide algumas palavras em partes menores. Por exemplo, Sylvain é dividido em 4 pedaços: S, ##yl, ##va, e ##in. No passo de pós-processamento, o pipeline satisfatoriamente reagrupa esses pedaços.

:::info
Experimente! Procure no [Hugging Face Model Hub](https://huggingface.co/models) por um modelo capaz de fazer o tageamento de partes do discurso (usualmente abreviado como POS) em inglês. O que o modelo prediz para a sentença no exemplo acima?
:::

## Responder perguntas

O pipeline question-answering responde perguntas usando informações dado um contexto:

<ColabButton href="https://colab.research.google.com/github/giseldo/cursos/blob/main/docs/pln/notebook/transformer.ipynb" />

```Python
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
```

```
{'score': 0.6385916471481323, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
```

Note que o pipeline funciona através da extração da informação dado um contexto; não gera uma resposta.

## Sumarização

Sumarização é uma tarefa de reduzir um texto em um texto menor enquanto pega toda (ou boa parte) dos aspectos importantes do texto referenciado. Aqui um exemplo:

```Python
from transformers import pipeline

summarizer = pipeline("summarization")
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)
```

```
[{'summary_text': ' America has changed dramatically during recent years . The '
                  'number of engineering graduates in the U.S. has declined in '
                  'traditional engineering disciplines such as mechanical, civil '
                  ', electrical, chemical, and aeronautical engineering . Rapidly '
                  'developing economies such as China and India, as well as other '
                  'industrial countries in Europe and Asia, continue to encourage '
                  'and advance engineering .'}]
```

Como a geração de texto, você pode especificar o tamanho máximo `max_length` ou mínimo `min_length` para o resultado.

## Tradução

Para tradução, você pode usar o modelo default se você der um par de idiomas no nome da tarefa (tal como "translation_en_to_fr", para traduzir inglês para francês), mas a maneira mais fácil é pegar o moddelo que você quiser e usa-lo no [Hugging Face Model Hub](https://huggingface.co/models). Aqui nós iremos tentar traduzir do Francês para o Inglês:

<ColabButton href="https://colab.research.google.com/github/giseldo/cursos/blob/main/docs/pln/notebook/transformer.ipynb" />

```Python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")
```

```
[{'translation_text': 'This course is produced by Hugging Face.'}]
Como a geração de texto e a sumarização, você pode especificar o tamanho máximo max_length e mínimo min_length para o resultado.
```

:::info
Experimente! Pesquise por modelos de tradução em outras línguas e experimente traduzir a sentença anterior em idiomas diferentes.
:::

Os pipelines mostrados até agora são em sua maioria para propósitos demonstrativos. Eles foram programados para tarefas específicas e não podem performar variações delas. No próximo capítulo, você aprenderá o que está por dentro da função pipeline() e como customizar seu comportamento.
