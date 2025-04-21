# Processamento de Linguagem Natural

## Introdução

O **Processamento de Linguagem Natural (PLN)** é um campo
intrinsecamente ligado à inteligência artificial, dedicando-se a equipar
computadores com a capacidade de analisar e compreender a linguagem
humana. No cenário da construção de *chatbots*, o PLN emerge como um
componente fundamental, atuando como o \"cérebro\" da aplicação
conversacional. Sua função primordial reside em processar a entrada
bruta do usuário, realizando a limpeza e a preparação dos dados textuais
para que o sistema possa interpretar a mensagem e tomar as ações
subsequentes apropriadas.

Em um espectro mais amplo, o PLN engloba uma vasta gama de tarefas que
transcendem a interação com *chatbots*. Ele se nutre de conceitos e
metodologias provenientes da ciência da computação, da linguística, da
matemática, da própria inteligência artificial, do aprendizado de
máquina e da psicologia. O objetivo central do PLN é, portanto,
manipular e analisar a linguagem natural, seja em sua forma escrita ou
falada, com o intuito de concretizar tarefas específicas e úteis. Este
processo multifacetado envolve a decomposição da linguagem em unidades
menores, a compreensão do seu significado intrínseco e a determinação da
resposta ou ação mais adequada.

## Entendimento de Linguagem Natural (ULN) como Subconjunto do PLN

O **Entendimento de Linguagem Natural (ULN)** é apresentado nas fontes
como um subdomínio específico dentro do universo mais vasto do PLN.
Enquanto o PLN abarca um conjunto diversificado de operações sobre a
linguagem, o ULN se concentra de maneira particular na habilidade da
máquina de apreender e interpretar a linguagem natural tal como ela é
comunicada pelos seres humanos. Em outras palavras, o ULN é o ramo do
PLN dedicado à extração de significado e à identificação da intenção por
trás do texto inserido pelo usuário. As aplicações do ULN são extensas e
incluem funcionalidades cruciais para *chatbots*, como a capacidade de
responder a perguntas, realizar buscas em linguagem natural, identificar
relações entre entidades, analisar o sentimento expresso no texto,
sumarizar informações textuais e auxiliar em processos de descoberta
legal.

## Técnicas Fundamentais de PLN para Chatbots

A construção de *chatbots* eficazes repousa sobre o emprego de diversas
técnicas de PLN, cada uma contribuindo para a capacidade do sistema de
interagir de forma inteligente com os usuários. As fontes detalham
algumas dessas técnicas essenciais:

### Tokenização

Este é o processo inicial de segmentar um texto em unidades menores
denominadas *tokens*, que podem ser palavras, pontuações ou símbolos. A
tokenização é um passo preparatório fundamental para qualquer análise
linguística subsequente.

Tokenizar não é só separar por espaços, mas também lidar com pontuações,
contrações e outros aspectos que podem afetar a análise. Por exemplo,
\"não é\" pode ser tokenizado como \[\"não\", \"é\"\] ou \[\"não\",
\"é\"\], dependendo do contexto e da abordagem adotada.

Um exemplo simples seria a frase \"Eu estou feliz.\", que seria
tokenizada em \[\"Eu\", \"estou\", \"feliz\", \".\"\]. Não
necessariamente uma palavra equivale a um token. Em alguns casos, como
em palavras compostas ou expressões idiomáticas, um único token pode
representar uma ideia ou conceito mais amplo. Por exemplo, \"São Paulo\"
poderia ser considerado um único token em vez de dois (\"São\" e
\"Paulo\").

Existem diferentes abordagens para tokenização, incluindo tokenização
baseada em regras, onde padrões específicos são definidos para
identificar tokens (geralmente utilizando expressão regular), e
tokenização baseada em aprendizado de máquina, onde algoritmos aprendem
a segmentar o texto com base em exemplos anteriores.

A tokenização pode ser feita de várias maneiras, dependendo do idioma e
do objetivo da análise. Em inglês, por exemplo, a tokenização pode ser
mais simples devido à estrutura gramatical, enquanto em idiomas como o
chinês, onde não há espaços entre as palavras, a tokenização pode ser
mais complexa.

### Marcação Morfossintática (POS Tagging)

Esta técnica consiste em atribuir a cada *token* em um texto uma
categoria gramatical, como substantivo, verbo, adjetivo, advérbio, etc..
A marcação POS é crucial para identificar entidades e compreender a
estrutura gramatical das frases. Por exemplo, na frase \"Eu estou
aprendendo como construir chatbots\", a marcação POS poderia identificar
\"Eu\" como um pronome (PRON), \"estou aprendendo\" como um verbo (VERB)
e \"chatbots\" como um substantivo (NOUN).

## Stemming e Lemmatização

Ambas as técnicas visam reduzir palavras flexionadas à sua forma base. O
*stemming* é um processo mais heurístico que remove sufixos, podendo nem
sempre resultar em uma palavra válida. Já a *lemmatização* é um processo
algorítmico que considera o significado da palavra para determinar seu
*lema*, ou seja, sua forma canônica. Por exemplo, a palavra \"correndo\"
poderia ser reduzida ao stem \"corr\" pelo *stemming* e ao lema
\"correr\" pela *lemmatização*. A lematização é geralmente preferível em
aplicações que exigem maior precisão semântica. A remoção de sufixos é
um objetivo comum dessas técnicas.

### Reconhecimento de Entidades Nomeadas (NER)

O NER é a tarefa de identificar e classificar entidades nomeadas em um
texto, como nomes de pessoas (PERSON), organizações (ORG), localizações
geográficas (GPE, LOC), datas (DATE), valores monetários (MONEY), etc..
Por exemplo, na frase \"Google tem sua sede em Mountain View,
Califórnia, com uma receita de 109.65 bilhões de dólares americanos\", o
NER identificaria \"Google\" como uma organização (ORG), \"Mountain
View\" e \"Califórnia\" como localizações geográficas (GPE) e \"109.65
bilhões de dólares americanos\" como um valor monetário (MONEY). Essa
capacidade é vital para que *chatbots* compreendam os detalhes
relevantes nas *utterances* dos usuários.

### Remoção de Palavras de Parada (Stop Words)

Palavras de parada são vocábulos de alta frequência que geralmente não
carregam muito significado contextual, como \"a\", \"o\", \"de\",
\"para\", \"que\". A remoção dessas palavras pode melhorar a eficácia de
certos algoritmos de PLN, focando nas palavras mais informativas do
texto.

### Análise de Dependências (Dependency Parsing)

Esta técnica examina as relações gramaticais entre as palavras em uma
frase, revelando a estrutura sintática e as dependências entre os
*tokens*. A análise de dependências pode ajudar a entender quem está
fazendo o quê a quem. Por exemplo, na frase \"Reserve um voo de
Bangalore para Goa\", a análise de dependências pode identificar
\"Bangalore\" e \"Goa\" como modificadores de \"voo\" através das
preposições \"de\" e \"para\", respectivamente, e \"Reserve\" como a
raiz da ação. Essa análise é útil para extrair informações sobre as
intenções do usuário, mesmo em frases mais complexas.

### Identificação de Grupos Nominais (Noun Chunks)

Esta técnica visa identificar sequências contínuas de palavras que atuam
como um sintagma nominal. Grupos nominais representam entidades ou
conceitos chave em uma frase. Um exemplo seria na frase \"Boston
Dynamics está se preparando para produzir milhares de cães robóticos\",
onde \"Boston Dynamics\" e \"milhares de cães robóticos\" seriam
identificados como grupos nominais.

### Busca por Similaridade

Utilizando vetores de palavras (*word embeddings*), como os gerados pelo
algoritmo GloVe, é possível calcular a similaridade semântica entre
palavras ou frases. Essa técnica permite que *chatbots* reconheçam que
palavras diferentes podem ter significados relacionados. Por exemplo,
\"carro\" e \"caminhão\" seriam considerados mais similares do que
\"carro\" e \"google\". Isso é útil para lidar com a variedade de
expressões que os usuários podem usar para expressar a mesma intenção.

### Expressões Regulares

São padrões de texto que podem ser usados para corresponder a sequências
específicas de caracteres. Embora não sejam uma técnica de PLN no mesmo
sentido que as outras, as expressões regulares são ferramentas poderosas
para identificar padrões em texto, como números de telefone, endereços
de e-mail ou formatos específicos de entrada.

### Classificação de Texto

Uma técnica de aprendizado de máquina que atribui um texto a uma ou mais
categorias predefinidas. No contexto de *chatbots*, a classificação de
texto é fundamental para a detecção de intenção, onde as categorias
representam as diferentes intenções do usuário. Algoritmos como o *Naïve
Bayes* são modelos estatísticos populares para essa tarefa, baseados no
teorema de Bayes e em fortes suposições de independência entre as
características. O treinamento desses classificadores requer um *corpus*
de dados rotulados, onde cada *utterance* (entrada do usuário) é
associada a uma intenção específica.

## Ferramentas e Bibliotecas de PLN Populares

- **spaCy**: Uma biblioteca de PLN de código aberto em Python e Cython,
  conhecida por sua velocidade e eficiência. O spaCy oferece APIs
  intuitivas e modelos pré-treinados para diversas tarefas de PLN,
  incluindo tokenização, POS tagging, lematização, NER e análise de
  dependências. Sua arquitetura é focada em desempenho para aplicações
  em produção.

- **NLTK (Natural Language Toolkit)**: Uma biblioteca Python fundamental
  para PLN, oferecendo uma ampla gama de ferramentas e recursos para
  tarefas como tokenização, stemming, POS tagging, análise sintática e
  NER. O NLTK é frequentemente utilizado para fins educacionais e de
  pesquisa.

- **CoreNLP (Stanford CoreNLP)**: Um conjunto de ferramentas de PLN
  robusto e amplamente utilizado, desenvolvido em Java. O CoreNLP
  oferece capacidades abrangentes de análise linguística, incluindo POS
  tagging, análise de dependências, NER e análise de sentimentos. Possui
  APIs para integração com diversas linguagens de programação, incluindo
  Python.

- **gensim**: Uma biblioteca Python especializada em modelagem de
  tópicos, análise de similaridade semântica e vetores de palavras. O
  gensim é particularmente útil para identificar estruturas semânticas
  em grandes coleções de texto.

- **TextBlob**: Uma biblioteca Python mais simples, construída sobre
  NLTK e spaCy, que fornece uma interface fácil de usar para tarefas
  básicas de PLN, como POS tagging, análise de sentimentos e correção
  ortográfica.

- **Rasa NLU**: Um componente de código aberto do framework Rasa para
  construir *chatbots*, focado em entendimento de linguagem natural.
  Rasa NLU permite treinar modelos personalizados para classificação de
  intenção e extração de entidades, oferecendo flexibilidade e controle
  sobre os dados.

## O Papel Crucial do PLN na Construção de Chatbots

No cerne da funcionalidade de um *chatbot* reside a sua capacidade de
compreender as mensagens dos usuários e responder de forma adequada. O
PLN desempenha um papel central nesse processo, permitindo que o
*chatbot*:

- **Detecte a Intenção do Usuário**: Identificar o objetivo por trás da
  mensagem do usuário é o primeiro passo crucial. Isso é frequentemente
  abordado como um problema de classificação de texto, onde o *chatbot*
  tenta classificar a *utterance* do usuário em uma das intenções
  predefinidas. As fontes mencionam o uso de técnicas de aprendizado de
  máquina, como o algoritmo *Naïve Bayes*, para construir esses
  classificadores. Plataformas como LUIS.ai e Rasa NLU simplificam
  significativamente o processo de treinamento e implantação desses
  modelos de intenção.

- **Extraia Entidades Relevantes**: Além da intenção geral, as mensagens
  dos usuários frequentemente contêm detalhes específicos, conhecidos
  como entidades, que são essenciais para atender à solicitação. Por
  exemplo, em \"Reserve um voo de Londres para Nova York amanhã\", a
  intenção é reservar um voo, e as entidades são a cidade de origem
  (\"Londres\"), a cidade de destino (\"Nova York\") e a data
  (\"amanhã\"). As técnicas de NER e os modelos de extração de entidades
  fornecidos por ferramentas como spaCy, NLTK, CoreNLP, LUIS.ai e Rasa
  NLU são fundamentais para identificar e extrair essas informações
  cruciais.

- **Processe Linguagem Variada e Informal**: Os usuários podem se
  comunicar com *chatbots* usando uma ampla gama de vocabulário,
  gramática e estilo, incluindo erros de digitação, abreviações e
  linguagem informal. As técnicas de PLN, como stemming, lematização e
  busca por similaridade, ajudam o *chatbot* a lidar com essa
  variabilidade e a compreender a essência da mensagem, mesmo que não
  seja expressa de forma perfeitamente gramatical.

- **Mantenha o Contexto da Conversa**: Em conversas mais longas, o
  significado de uma *utterance* pode depender do que foi dito
  anteriormente. Embora as fontes não detalhem profundamente o
  gerenciamento de contexto, subentendem que o PLN, juntamente com
  outras técnicas de gerenciamento de diálogo, contribui para a
  capacidade do *chatbot* de lembrar informações e entender referências
  implícitas.

## PLN na Arquitetura de Chatbots

A arquitetura típica de um *chatbot* envolve uma camada de processamento
de linguagem natural (NLP/NLU engine) que recebe a entrada de texto do
usuário. Essa camada é responsável por realizar as tarefas de PLN
mencionadas anteriormente: tokenização, análise morfossintática,
extração de entidades, detecção de intenção, etc.. O resultado desse
processamento é uma representação estruturada da mensagem do usuário,
que pode ser entendida pela lógica de negócios do *chatbot*.

Com base nessa representação estruturada, um motor de decisão (*decision
engine*) no *chatbot* pode então corresponder a intenção do usuário a
fluxos de trabalho preconfigurados ou a regras de negócio específicas.
Em alguns casos, a geração de linguagem natural (NLG), outro subcampo do
PLN, é utilizada para formular a resposta do *chatbot* ao usuário.

## Desafios da PLN

- Geração de Texto coerente

- Sintaxe e gramática

- semântica

- Contexto

- Ambiguidade

A Geração de Texto coerente é um desafio porque envolve não apenas a
escolha de palavras, mas também a construção de frases que façam sentido
no contexto da conversa. A sintaxe e gramática são importantes para
garantir que o texto gerado seja gramaticalmente correto e
compreensível. A semântica se refere ao significado das palavras e
frases, e é importante para garantir que o texto gerado transmita a
mensagem correta. O contexto é importante para entender o que foi dito
anteriormente na conversa e como isso afeta a resposta atual. A
ambiguidade pode surgir quando uma palavra ou frase tem múltiplos
significados, tornando difícil para o modelo determinar qual
interpretação é a correta.
