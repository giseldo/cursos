# Transformers Quickstart

Essa é uma traduçao (com pequenos ajustes) do material [transformers quicktour](https://huggingface.co/docs/transformers/quicktour) do Hugging Face.

## Introdução

Este post mostrará como usar o ```pipeline()``` para inferência, como carregar um modelo pré-treinado, um pré-processador e treinar um modelo com **PyTorch**.

Antes de começar, certifique-se de ter todas as bibliotecas necessárias instaladas:

```bash
pip install transformers datasets evaluate accelerate torch
```

O ```pipeline()``` é a maneira mais fácil e rápida de usar um modelo pré-treinado para inferência.
Você pode usar o ```pipeline()``` pronto para uso para muitas Tarefas (Tabela 1) em diferentes modalidades, algumas das quais são mostradas na tabela abaixo:

<center>Tabela 1 - Tarefas possíveis com o pipeline do Transformers.</center>

|Descriçao da Tarefa|Identificador do Pipeline|
|----|----|
|Classificação de texto|pipeline(task=“sentiment-analysis”)|
|Geração de Texto|pipeline(task=“text-generation”)|
|reconhecimento automático de fala|pipeline(task=“automatic-speech-recognition”)|

## Exemplo análise de sentimento

Comece criando uma instância de ```pipeline()``` e especificando uma tarefa para a qual você deseja usá-lo.
O ```pipeline()``` baixa e armazena em cache um modelo pré-treinado padrão e um tokenizador para análise de sentimento.
Agora você pode usar o classificador no seu texto de destino.
Neste guia, você usará o ```pipeline()``` para análise de sentimentos como um exemplo:

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
classifier("We are very happy to show you the Transformers library.")
```

```
[{'label': 'POSITIVE', 'score': 0.9997795224189758}]
```

Se você tiver mais de uma entrada, passe suas entradas como uma lista para o ```pipeline()``` para retornar uma lista de dicionários:

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
results = classifier(["We are very happy to show you the Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
```

```
    label: POSITIVE, with score: 0.9998
    label: NEGATIVE, with score: 0.5309
```

## Exemplo reconhecimento automático de fala

O ```pipeline()``` também pode iterar um conjunto de dados inteiro para qualquer tarefa que você desejar. Para este exemplo, vamos escolher o pipeline **reconhecimento automático de fala** utilizando o modelo [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)

Instale a dependência:

```bash
pip install librosa soundfile 
```

Carregue um conjunto de dados de áudio que você gostaria de iterar.
Por exemplo, carregue o conjunto de dados [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14)

```python
import torch
from transformers import pipeline
from datasets import load_dataset, Audio

speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# carregando o conjunto de dados MInDS-14
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")

# Você precisa ter certeza de que a taxa de amostragem do conjunto de dados 
# corresponde à taxa de amostragem em que facebook/wav2vec2-base-960h foi treinado:
dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))

# Os arquivos de áudio são automaticamente carregados e reamostrados ao chamar a coluna "audio".
# Extraia os arrays de forma de onda bruta (raw waveform) das primeiras 4 amostras e passe-os como uma lista para o pipeline:
result = speech_recognizer(dataset[:4]["audio"])
print([d["text"] for d in result])
```

```
['I WOULD LIKE TO SET UP A JOINT ACCOUNT WITH MY PARTNER HOW DO I PROCEED WITH DOING THAT', 
"FONDERING HOW I'D SET UP A JOIN TO HELL T WITH MY WIFE AND WHERE THE AP MIGHT BE", 
... 
'HOW DO I FURN A JOINA COUT']
```

Mais sobre datasets pode ser encontrado em [Hugging Face Dataset Quick Tour](https://huggingface.co/docs/datasets/quickstart)

Para conjuntos de dados maiores onde as entradas são grandes (como fala ou visão), você desejará passar um gerador em vez de uma lista para carregar todas as entradas na memória. Dê uma olhada na referência da [API do pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) para obter mais informações.

## Use outro modelo e tokenizer no pipeline

O ```pipeline()``` pode acomodar qualquer modelo (model) do [Hub](https://huggingface.co/models), facilitando a adaptação do ```pipeline()``` para outros casos de uso. Por exemplo, se você quiser um modelo capaz de lidar com texto em francês, encontre o nome do modelo realizando uma busca no [Hub](https://huggingface.co/models). Faça o filtro por task="Text classification", Language="fr" e sorte="liked" para encontrar um modelo apropriado. O resultado do [filtro anterior](https://huggingface.co/models?pipeline_tag=text-classification&language=fr&sort=likes) retorna o modelo BERT [bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) ajustado (finetuned) para **análise de sentimento** que você pode usar para textos em francês. Este modelo ajustado retorna 1 a 5 estrelas e foi treinado com reviews de produtos. Segue um exemplo de uso do modelo encontrado (BERT) para análise de sentimento de um texto em inglês.

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model='nlptown/bert-base-multilingual-uncased-sentiment')
print(classifier("We are very happy to show you the Transformers library."))
```

```
[{'label': '5 stars', 'score': 0.7495927214622498}]
```

É possível informar além do modelo para o ```pipeline``` um tokenizer diferente. Vamos identificar qual o tokenizer associado a determinado modelo, e informá-lo no parâmetro do ```pipeline```

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print(classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers."))
```

```
[{'label': '5 stars', 'score': 0.7272651791572571}]
```

Se não conseguir encontrar um modelo para seu caso de uso, você precisará ajustar um modelo pré-treinado em seus dados. Dê uma olhada no [tutorial de ajuste fino](https://huggingface.co/docs/transformers/training) para saber como. Por fim, depois de ajustar seu modelo pré-treinado, considere [compartilhar o modelo](https://huggingface.co/docs/transformers/model_sharing) com a comunidade no Hub para democratizar o aprendizado de máquina para todos!
