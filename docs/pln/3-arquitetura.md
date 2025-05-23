# Como os Transformers trabalham?

::: info
esta é uma versão com pequenas alterações do material original, consulte o material original, criado pela equipe do Hugging Face [aqui](https://moon-ci-docs.huggingface.co/learn/nlp-course/pr_740/pt/chapter1/4?fw=pt).
:::

Nessa seção, nós olharemos para o alto nível de arquitetura dos modelos Transformers.

::: info
A arquitetura tranformer não é a mesma coisa da api `transformer` da seção anterior.
:::

## Um pouco da história dos Transformers

Aqui alguns pontos de referência na (pequena) história dos modelos Transformers:

<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_chrono.svg" alt="A brief chronology of Transformers models.">

A [arquitetura Transformer](https://arxiv.org/abs/1706.03762) foi introduzida em Junho de 2017. O foco de pesquisa original foi para tarefas de tradução. 

::: info Bate papo com duas pessoas sobre Paper Attention is all you need gerado automaticamente pelo notebookLM
<audio controls>
    <source src="./audio/attention_is_all_you_need.wav" type="audio/mpeg">
    Seu navegador não suporta o elemento de áudio.
</audio>
:::

Isso foi seguido pela introdução de muitos modelos influentes, incluindo:

- **Junho de 2018**: [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), o primeiro modelo Transformer pré-treinado, usado para ajuste-fino em várias tarefas de NLP e obtendo resultados estado-da-arte

- **Outubro de 2018**: [BERT](https://arxiv.org/abs/1810.04805), outro grande modelo pré-treinado, esse outro foi designado para produzir melhores resumos de sentenças(mais sobre isso no próximo capítulo!)

- **Fevereiro de 2019**: [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), uma melhor (e maior) versão da GPT que não foi imediatamente publicizado o seu lançamento devido a alegações de preocupações éticas por parte da OpenAI.

- **Outubro de 2019**: [DistilBERT](https://arxiv.org/abs/1910.01108), uma versão destilada do BERT que é 60% mais rápidam 40% mais leve em memória, e ainda retém 97% da performance do BERT

- **Outubro de 2019**: [BART](https://arxiv.org/abs/1910.13461) e [T5](https://arxiv.org/abs/1910.10683), dois grandes modelos pré-treinados usando a mesma arquitetura do modelo original Transformer (os primeiros a fazerem até então)

- **Maio de 2020**, [GPT-3](https://arxiv.org/abs/2005.14165), uma versão ainda maior da GPT-2 que é capaz de performar bem em uma variedade de tarefas sem a necessidade de ajuste-fino (chamado de aprendizagem_zero-shot_)

Esta lista está longe de ser abrangente e destina-se apenas a destacar alguns dos diferentes tipos de modelos de Transformers. Em linhas gerais, eles podem ser agrupados em três categorias:

- GPT-like (também chamados de modelos Transformers _auto-regressivos_)
- BERT-like (também chamados de modelos Transformers _auto-codificadores_) 
- BART/T5-like (também chamados de modelos Transformers _sequence-to-sequence_)

Vamos mergulhar nessas famílias com mais profundidade mais adiante

## Transformers são modelos de linguagem

![alt text](fig/apreendizado.png)

Todos os modelos de Transformer mencionados acima (GPT, BERT, BART, T5, etc.) foram treinados como *modelos de linguagem*. Isso significa que eles foram treinados em grandes quantidades de texto bruto de forma auto-supervisionada. O aprendizado autossupervisionado é um tipo de treinamento no qual o objetivo é calculado automaticamente a partir das entradas do modelo. Isso significa que os humanos não são necessários para rotular os dados!

Este tipo de modelo desenvolve uma compreensão estatística da linguagem em que foi treinado, mas não é muito útil para tarefas práticas específicas. Por causa disso, o modelo geral pré-treinado passa por um processo chamado *aprendizagem de transferência*. Durante esse processo, o modelo é ajustado de maneira supervisionada - ou seja, usando rótulos anotados por humanos - em uma determinada tarefa.

Um exemplo de tarefa é prever a próxima palavra em uma frase depois de ler as *n* palavras anteriores. Isso é chamado de *modelagem de linguagem causal* porque a saída depende das entradas passadas e presentes, mas não das futuras.

<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/causal_modeling.svg" alt="Example of causal language modeling in which the next word from a sentence is predicted.">

Outro exemplo é a *modelagem de linguagem mascarada*, na qual o modelo prevê uma palavra mascarada na frase.

<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/masked_modeling.svg" alt="Example of masked language modeling in which a masked word from a sentence is predicted.">
## Transformers são modelos grandes

Além de alguns outliers (como o DistilBERT), a estratégia geral para obter melhor desempenho é aumentar os tamanhos dos modelos, bem como a quantidade de dados em que são pré-treinados.

<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/model_parameters.png" alt="Number of parameters of recent Transformers models" width="90%">

Infelizmente, treinar um modelo, especialmente um grande, requer uma grande quantidade de dados. Isso se torna muito caro em termos de tempo e recursos de computação. Até se traduz em impacto ambiental, como pode ser visto no gráfico a seguir.

<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/carbon_footprint.svg" alt="The carbon footprint of a large language model.">

E isso mostra um projeto para um modelo (muito grande) liderado por uma equipe que tenta conscientemente reduzir o impacto ambiental do pré-treinamento. Os gastos de executar muitos testes para obter os melhores hiperparâmetros seria ainda maior.

Imagine se cada vez que uma equipe de pesquisa, uma organização estudantil ou uma empresa quisesse treinar um modelo, o fizesse do zero. Isso levaria a custos globais enormes e desnecessários!

É por isso que compartilhar modelos de linguagem é fundamental: compartilhar os pesos treinados e construir em cima dos pesos já treinados reduz o custo geral de computação e os gastos de carbono da comunidade.


## Transferência de Aprendizagem

![alt text](fig/preajsute.png)

*Pré-treinamento* é o ato de treinar um modelo do zero: os pesos são inicializados aleatoriamente e o treinamento começa sem nenhum conhecimento prévio.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/pretraining.svg" alt="The pretraining of a language model is costly in both time and money.">
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/pretraining-dark.svg" alt="The pretraining of a language model is costly in both time and money.">
</div>

Esse pré-treinamento geralmente é feito em grandes quantidades de dados. Portanto, requer um corpus de dados muito grande e o treinamento pode levar várias semanas.

*Ajuste fino*, por outro lado, é o treinamento feito **após** um modelo ter sido pré-treinado. Para realizar o ajuste fino, primeiro você adquire um modelo de linguagem pré-treinado e, em seguida, realiza treinamento adicional com um conjunto de dados específico para sua tarefa. Espere - por que não simplesmente treinar diretamente para a tarefa final? Existem algumas razões:

*  O modelo pré-treinado já foi treinado em um conjunto de dados que possui algumas semelhanças com o conjunto de dados de ajuste fino. O processo de ajuste fino é, portanto, capaz de aproveitar o conhecimento adquirido pelo modelo inicial durante o pré-treinamento (por exemplo, com problemas de NLP, o modelo pré-treinado terá algum tipo de compreensão estatística da linguagem que você está usando para sua tarefa).
* Como o modelo pré-treinado já foi treinado com muitos dados, o ajuste fino requer muito menos dados para obter resultados decentes.
*  Pela mesma razão, a quantidade de tempo e recursos necessários para obter bons resultados são muito menores.

Por exemplo, pode-se alavancar um modelo pré-treinado treinado no idioma inglês e depois ajustá-lo em um corpus arXiv, resultando em um modelo baseado em ciência/pesquisa. O ajuste fino exigirá apenas uma quantidade limitada de dados: o conhecimento que o modelo pré-treinado adquiriu é "transferido", daí o termo *aprendizagem de transferência*.

<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/finetuning.svg" alt="The fine-tuning of a language model is cheaper than pretraining in both time and money.">

O ajuste fino de um modelo, portanto, tem menores custos de tempo, dados, financeiros e ambientais. Também é mais rápido e fácil iterar em diferentes esquemas de ajuste fino, pois o treinamento é menos restritivo do que um pré-treinamento completo.

Esse processo também alcançará melhores resultados do que treinar do zero (a menos que você tenha muitos dados), e é por isso que você deve sempre tentar alavancar um modelo pré-treinado - um o mais próximo possível da tarefa que você tem em mãos - e  então fazer seu ajuste fino.

## Arquitetura geral

Nesta seção, veremos a arquitetura geral do modelo Transformer.

## Introdução

O modelo é principalmente composto por dois blocos:

* **Codificador (esquerda)**: O codificador recebe uma entrada e constrói uma representação dela (seus recursos). Isso significa que o modelo é otimizado para adquirir entendimento da entrada.
* **Decodificador (à direita)**: O decodificador usa a representação do codificador (recursos) junto com outras entradas para gerar uma sequência de destino. Isso significa que o modelo é otimizado para gerar saídas.


<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_blocks.svg" alt="Architecture of a Transformers models">

Cada uma dessas partes pode ser usada de forma independente, dependendo da tarefa:

* **Modelos somente de codificador**: bom para tarefas que exigem compreensão da entrada, como classificação de sentença e reconhecimento de entidade nomeada.
* **Modelos somente decodificadores**: bom para tarefas generativas, como geração de texto.
* **Modelos de codificador-decodificador** ou **modelos de sequência a sequência**: bom para tarefas generativas que exigem uma entrada, como tradução ou resumo. (tarefas sequence to sequence)

Vamos mergulhar nessas arquiteturas de forma independente em seções posteriores.

## Camadas de Atenção

Uma característica chave dos modelos Transformer é que eles são construídos com camadas especiais chamadas *camadas de atenção*. Na verdade, o título do artigo que apresenta a arquitetura do Transformer era ["Atenção é tudo que você precisa"](https://arxiv.org/abs/1706.03762)! Exploraremos os detalhes das camadas de atenção posteriormente no curso; por enquanto, tudo o que você precisa saber é que essa camada dirá ao modelo para prestar atenção específica a certas palavras na frase que você passou (e mais ou menos ignorar as outras) ao lidar com a representação de cada palavra.

Para contextualizar, considere a tarefa de traduzir o texto do português para o francês. Dada a entrada "Você gosta deste curso", um modelo de tradução precisará atender também à palavra adjacente "Você" para obter a tradução adequada para a palavra "gosta", pois em francês o verbo "gostar" é conjugado de forma diferente dependendo o sujeito. O resto da frase, no entanto, não é útil para a tradução dessa palavra. Na mesma linha, ao traduzir "deste" o modelo também precisará prestar atenção à palavra "curso", pois "deste" traduz-se de forma diferente dependendo se o substantivo associado é masculino ou feminino. Novamente, as outras palavras na frase não importarão para a tradução de "deste". Com frases mais complexas (e regras gramaticais mais complexas), o modelo precisaria prestar atenção especial às palavras que podem aparecer mais distantes na frase para traduzir adequadamente cada palavra.

O mesmo conceito se aplica a qualquer tarefa associada à linguagem natural: uma palavra por si só tem um significado, mas esse significado é profundamente afetado pelo contexto, que pode ser qualquer outra palavra (ou palavras) antes ou depois da palavra que está sendo estudada.

Agora que você tem uma ideia do que são as camadas de atenção, vamos dar uma olhada mais de perto na arquitetura do Transformer.

## A arquitetura original

![alt text](fig/codencode.png)

A arquitetura Transformer foi originalmente projetada para tradução. Durante o treinamento, o codificador recebe entradas (frases) em um determinado idioma, enquanto o decodificador recebe as mesmas frases no idioma de destino desejado. No codificador, as camadas de atenção podem usar todas as palavras em uma frase (já que, como acabamos de ver, a tradução de uma determinada palavra pode ser dependente do que está depois e antes dela na frase). O decodificador, no entanto, funciona sequencialmente e só pode prestar atenção nas palavras da frase que ele já traduziu (portanto, apenas as palavras anteriores à palavra que está sendo gerada no momento). Por exemplo, quando previmos as três primeiras palavras do alvo traduzido, as entregamos ao decodificador que então usa todas as entradas do codificador para tentar prever a quarta palavra.

Para acelerar as coisas durante o treinamento (quando o modelo tem acesso às frases alvo), o decodificador é alimentado com todo o alvo, mas não é permitido usar palavras futuras (se teve acesso à palavra na posição 2 ao tentar prever a palavra na posição 2, o problema não seria muito difícil!). Por exemplo, ao tentar prever a quarta palavra, a camada de atenção só terá acesso às palavras nas posições 1 a 3.

A arquitetura original do Transformer ficou assim, com o codificador à esquerda e o decodificador à direita:

<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers.svg" alt="Architecture of a Transformers models">

Observe que a primeira camada de atenção em um bloco decodificador presta atenção a todas as entradas (passadas) do decodificador, mas a segunda camada de atenção usa a saída do codificador. Ele pode, assim, acessar toda a frase de entrada para melhor prever a palavra atual. Isso é muito útil, pois diferentes idiomas podem ter regras gramaticais que colocam as palavras em ordens diferentes, ou algum contexto fornecido posteriormente na frase pode ser útil para determinar a melhor tradução de uma determinada palavra.

A *máscara de atenção* também pode ser usada no codificador/decodificador para evitar que o modelo preste atenção a algumas palavras especiais - por exemplo, a palavra de preenchimento especial usada para fazer com que todas as entradas tenham o mesmo comprimento ao agrupar frases.

##  Arquiteturas vs. checkpoints

À medida que nos aprofundarmos nos modelos do Transformer neste curso, você verá menções a *arquiteturas* e *checkpoints*, bem como *modelos*. Todos esses termos têm significados ligeiramente diferentes:

* **Arquitetura**: Este é o esqueleto do modelo -- a definição de cada camada e cada operação que acontece dentro do modelo.
* **Checkpoints**: Esses são os pesos que serão carregados em uma determinada arquitetura.
* **Modelos**: Este é um termo abrangente que não é tão preciso quanto "arquitetura" ou "checkpoint": pode significar ambos. Este curso especificará *arquitetura* ou *checkpoint* quando for necessário reduzir a ambiguidade.

Por exemplo, BERT é uma arquitetura enquanto `bert-base-cased`, um conjunto de pesos treinados pela equipe do Google para a primeira versão do BERT, é um checkpoint. No entanto, pode-se dizer "o modelo BERT" e "o modelo `bert-base-cased`".