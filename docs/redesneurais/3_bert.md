# Modelo base BERT (uncased)

## Introdução

:::info
uncase = sem caixa alta, ou seja tudo minúsculo
:::

O `bert base uncased model` é um mdelo `pré-treinado (pre-training)`  em inglês usando o objetivo de `modelagem de linguagem mascarada (mask laguage modeling - MLM)`. Ele foi apresentado neste [artigo](https://arxiv.org/abs/1810.04805) e lançado pela primeira vez neste [repositório](https://github.com/google-research/bert). Este modelo é uncased. Por exemplo, não faz distinção entre inglês e Inglês.

```mermaid
graph LR
  A[Pré-processamento] --> B[Tokenização WordPiece]
  B --> C[Máscara de 15% dos tokens]
  C --> D[Substituição por [MASK], aleatório ou mantém]
  D --> E[Pré-treinamento]
  E --> F[MLM (Modelagem de Linguagem Mascarada)]
  E --> G[NSP (Previsão da Próxima Frase)]
  F --> H[Representação Bidirecional]
  G --> H
  H --> I[Ajuste fino para tarefas específicas]
```

## Descrição do modelo

O BERT é um modelo de `tranformer` pré-treinado em um grande corpus de dados em inglês de forma autossupervisionada. Isso significa que ele foi pré-treinado apenas com os textos brutos, sem a necessidade de qualquer tipo de rotulagem humana (e é por isso que ele pode usar muitos dados disponíveis publicamente), com um processo automático para gerar entradas e rótulos a partir desses textos. Mais precisamente, ele foi pré-treinado com dois objetivos:

- Modelagem de linguagem mascarada (MLM): pegando uma frase, o modelo mascara aleatoriamente 15% das palavras na entrada e, em seguida, executa a frase mascarada inteira no modelo e precisa prever as palavras mascaradas. Isso é diferente das redes neurais recorrentes (RNNs) tradicionais, que geralmente veem as palavras uma após a outra, ou de modelos autorregressivos como o GPT, que mascara internamente os tokens futuros. Isso permite que o modelo aprenda uma representação bidirecional da frase.

- Previsão da próxima frase (NSP): o modelo concatena duas frases mascaradas como entradas durante o pré-treinamento. Às vezes, elas correspondem a frases que estavam próximas uma da outra no texto original, às vezes não. O modelo então precisa prever se as duas frases estavam uma após a outra ou não.
Dessa forma, o modelo aprende uma representação interna do idioma inglês que pode então ser usada para extrair recursos úteis para tarefas posteriores: se você tiver um conjunto de dados de frases rotuladas, por exemplo, poderá treinar um classificador padrão usando os recursos produzidos pelo modelo BERT como entradas.

Variações do modelo
O BERT foi originalmente lançado em versões base e grande, para texto de entrada com e sem caixa. Os modelos sem caixa também removem os marcadores de acento.
Versões em chinês e multilíngue, com e sem caixa, surgiram logo em seguida.
O pré-processamento modificado com mascaramento de palavras inteiras substituiu o mascaramento de subparte em um trabalho subsequente, com o lançamento de dois modelos.
Outros 24 modelos menores são lançados posteriormente.

O histórico detalhado do lançamento pode ser encontrado no arquivo readme google-research/bert no github.

Modelo	#parâmetros	Linguagem
bert-base-uncased	110 milhões	Inglês
bert-large-uncased	340 milhões	Inglês
bert-base-cased	110 milhões	Inglês
bert-large-cased	340 milhões	Inglês
bert-base-chinese	110 milhões	chinês
bert-base-multilingual-cased	110 milhões	Múltiplos
bert-large-uncased-whole-word-masking	340 milhões	Inglês
bert-large-cased-whole-word-masking	340 milhões	Inglês
Usos pretendidos e limitações
Você pode usar o modelo bruto para modelagem de linguagem mascarada ou previsão da próxima frase, mas ele se destina principalmente a ajustes finos em uma tarefa posterior. Consulte o hub de modelos para procurar versões ajustadas de uma tarefa do seu interesse.

Observe que este modelo visa principalmente o ajuste fino em tarefas que usam a frase inteira (potencialmente mascarada) para tomar decisões, como classificação de sequências, classificação de tokens ou resposta a perguntas. Para tarefas como geração de texto, você deve considerar um modelo como o GPT2.

Como usar
Você pode usar este modelo diretamente com um pipeline para modelagem de linguagem mascarada:

```python
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
unmasker("Hello I'm a [MASK] model.")
```

```bash
[{'sequence': "[CLS] hello i'm a fashion model. [SEP]",
  'score': 0.1073106899857521,
  'token': 4827,
  'token_str': 'fashion'},
 {'sequence': "[CLS] hello i'm a role model. [SEP]",
  'score': 0.08774490654468536,
  'token': 2535,
  'token_str': 'role'},
 {'sequence': "[CLS] hello i'm a new model. [SEP]",
  'score': 0.05338378623127937,
  'token': 2047,
  'token_str': 'new'},
 {'sequence': "[CLS] hello i'm a super model. [SEP]",
  'score': 0.04667217284440994,
  'token': 3565,
  'token_str': 'super'},
 {'sequence': "[CLS] hello i'm a fine model. [SEP]",
  'score': 0.027095865458250046,
  'token': 2986,
  'token_str': 'fine'}]
```

Veja como usar este modelo para obter as características de um determinado texto no PyTorch:

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

e no TensorFlow:

```python
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

## Limitações e viés

Mesmo que os dados de treinamento usados ​​para este modelo possam ser caracterizados como bastante neutros, este modelo pode ter previsões tendenciosas:

```python
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
unmasker("The man worked as a [MASK].")
```

```bash
[{'sequence': '[CLS] the man worked as a carpenter. [SEP]',
  'score': 0.09747550636529922,
  'token': 10533,
  'token_str': 'carpenter'},
 {'sequence': '[CLS] the man worked as a waiter. [SEP]',
  'score': 0.0523831807076931,
  'token': 15610,
  'token_str': 'waiter'},
 {'sequence': '[CLS] the man worked as a barber. [SEP]',
  'score': 0.04962705448269844,
  'token': 13362,
  'token_str': 'barber'},
 {'sequence': '[CLS] the man worked as a mechanic. [SEP]',
  'score': 0.03788609802722931,
  'token': 15893,
  'token_str': 'mechanic'},
 {'sequence': '[CLS] the man worked as a salesman. [SEP]',
  'score': 0.037680890411138535,
  'token': 18968,
  'token_str': 'salesman'}]
```

```python
unmasker("The woman worked as a [MASK].")
```

```bash
[{'sequence': '[CLS] the woman worked as a nurse. [SEP]',
  'score': 0.21981462836265564,
  'token': 6821,
  'token_str': 'nurse'},
 {'sequence': '[CLS] the woman worked as a waitress. [SEP]',
  'score': 0.1597415804862976,
  'token': 13877,
  'token_str': 'waitress'},
 {'sequence': '[CLS] the woman worked as a maid. [SEP]',
  'score': 0.1154729500412941,
  'token': 10850,
  'token_str': 'maid'},
 {'sequence': '[CLS] the woman worked as a prostitute. [SEP]',
  'score': 0.037968918681144714,
  'token': 19215,
  'token_str': 'prostitute'},
 {'sequence': '[CLS] the woman worked as a cook. [SEP]',
  'score': 0.03042375110089779,
  'token': 5660,
  'token_str': 'cook'}]
```

Esse viés também afetará todas as versões refinadas deste modelo.

 
## Dados de treinamento

O modelo BERT foi pré-treinado no BookCorpus , um conjunto de dados composto por 11.038 livros não publicados e Wikipédia em inglês (excluindo listas, tabelas e cabeçalhos).

Procedimento de treinamento
Pré-processamento
Os textos são escritos em minúsculas e tokenizados usando o WordPiece e um vocabulário de 30.000 palavras. As entradas do modelo são então do tipo:

```bash
[CLS] Sentence A [SEP] Sentence B [SEP]
```

Com probabilidade de 0,5, as sentenças A e B correspondem a duas sentenças consecutivas no corpus original e, nos demais casos, é outra sentença aleatória no corpus. Observe que o que é considerado uma sentença aqui é um trecho consecutivo de texto, geralmente maior do que uma única sentença. A única restrição é que o resultado com as duas "sentenças" tem um comprimento combinado inferior a 512 tokens.

Os detalhes do procedimento de mascaramento para cada frase são os seguintes:

15% dos tokens são mascarados.
Em 80% dos casos, os tokens mascarados são substituídos por [MASK].
Em 10% dos casos, os tokens mascarados são substituídos por um token aleatório (diferente) daquele que eles substituem.
Nos 10% de casos restantes, os tokens mascarados são deixados como estão.

## Pré-treinamento

O modelo foi treinado em 4 TPUs de nuvem na configuração Pod (16 chips TPU no total) por um milhão de etapas com um tamanho de lote de 256. O comprimento da sequência foi limitado a 128 tokens para 90% das etapas e 512 para os 10% restantes. O otimizador utilizado é o Adam, com uma taxa de aprendizado de 1e-4.

<!-- β
1
=
0,9
β 
1
​
 =0,9e
β
2
=
0,999
β 
2 -->
​
 =0,999, uma queda de peso de 0,01, aquecimento da taxa de aprendizagem para 10.000 passos e queda linear da taxa de aprendizagem depois.

Resultados da avaliação
Quando ajustado em tarefas posteriores, este modelo alcança os seguintes resultados:

Resultados do teste de cola:

<!-- Tarefa	MNLI-(m/mm)	QQP	QNLI	SST-2	Cola	STS-B	MRPC	RTE	Média
84,6/83,4	71,2	90,5	93,5	52,1	85,8	88,9	66,4	79,6
Informações de entrada e citação do BibTeX -->