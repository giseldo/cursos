# Introdução

## Definição

A regressão linear (simples/múltipla ou Logística) é uma técnica estatística utilizada para modelar a relação entre uma variável dependente contínua e uma ou mais variáveis independentes.
Caso haja somente uma variável independente é chamado de regressão linear simples, com mais de uma variável independente será chamado de regressão múltipla.
O objetivo principal é encontrar a melhor linha reta que descreve a relação entre as variáveis. Uma forma de calcular esta reta é minimizar a soma dos quadrados das diferenças entre os valores observados e os valores previstos.

Matematicamente, a equação da regressão linear pode ser expressa como:

$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n + \epsilon$$

onde:

- $y$ é a variável dependente,

- $x_1, x_2, \ldots, x_n$ são as variáveis independentes,

- $\beta_0$ é o intercepto,

- $\beta_1, \beta_2, \ldots, \beta_n$ são os coeficientes das
    variáveis independentes,

- $\epsilon$ é o termo de erro.

{{< alert text="O que é uma variável dependente e independente serão explicado nas seções seguintes." />}}

## História e Evolução

O conceito de regressão linear começou com _Sir Francis Galton_, que introduziu o termo "regressão" em um estudo sobre a hereditariedade da altura em 1877.
Ele observou que a altura dos filhos tendia a regredir em direção à média da altura dos pais, um fenômeno que ele chamou de
"regressão para a média" [[1]](#1).

Posteriormente, _Karl Pearson_, um pioneiro da estatística moderna, expandiu o trabalho de Galton e formalizou o método de regressão linear.
Em 1896, Pearson introduziu a técnica de mínimos quadrados para estimar os coeficientes de regressão, que se tornou a base para o método de regressão linear [[2]](#2).

[//]: # "Este parágrafo foi alterado incluir no PDF"
Em seguida, _Ronald A. Fisher_, um dos fundadores da estatística moderna, contribuiu significativamente para a regressão linear durante as décadas de 1920 e 1930.
Ele desenvolveu a análise de variância (ANOVA), uma forma de analisar os resultados de experimentos que envolvem múltiplos fatores e determinar se as diferenças observadas entre grupos ou tratamentos são estatisticamente significativas.
Isso ajudou a estender a regressão linear para incluir múltiplas variáveis independentes [[3]](#3).
_Fisher_ também introduziu o conceito de máxima verossimilhança, que aprimorou os métodos de estimação  e parâmetros.

Com o avanço dos computadores e das técnicas de computação nos anos 1960 e 1970, a regressão linear tornou-se uma ferramenta essencial em diversas áreas. Hoje, a regressão linear é amplamente utilizada em aprendizagem de máquina como um ponto de partida para o desenvolvimento de modelos preditivos.

A introdução de métodos computacionais avançados e o surgimento de softwares estatísticos, tais como R e Python, tornaram a aplicação da regressão linear mais acessível e poderosa.
Isso permitiu o processamento de grandes volumes de dados e a aplicação de regressão linear em uma ampla gama de disciplinas [[4]](#4) [[5]](#5).
Veja na **Figura 1** uma linha do tempo dessa evolução.

**Figura 1** - Linha do Tempo
<img src="/linhadotempo.png" alt="" srcset=""></br>
Fonte: O autor.

{{% alert context="light" %}}
O Python foi criado por Guido van Rossum e lançado pela primeira vez em 1991. Van Rossum começou a desenvolver Python no final dos anos 1980 como um sucessor de uma linguagem chamada ABC.
A ideia era criar uma linguagem de programação que fosse fácil de entender e de usar, com uma sintaxe clara e legível. O nome Python foi escolhido como uma referência ao grupo de comédia britânico Monty Python, do qual Van Rossum era fã.
Desde o seu lançamento, Python tem evoluído significativamente, com várias versões lançadas ao longo dos anos, incluindo as séries Python 2.x e Python 3.x, sendo esta última a mais atual e recomendada para novos projetos.
{{% /alert %}}

{{% alert context="light" %}}
A linguagem de programação R surgiu em meados da década de 1990.
Ela foi criada por Ross Ihaka e Robert Gentleman, dois estatísticos da Universidade de Auckland, na Nova Zelândia. O desenvolvimento inicial do R começou em 1992, e a primeira versão pública foi lançada em 1995. R foi projetada como uma linguagem para estatística e análise de dados, fortemente influenciada pela linguagem S, que foi desenvolvida
anteriormente nos laboratórios da Bell.
Uma das principais vantagens do R é a sua capacidade de fornecer uma ampla gama de ferramentas estatísticas e gráficas, tornando-o popular entre estatísticos, cientistas de dados e pesquisadores em várias disciplinas.
Desde o seu lançamento, R tem se expandido com contribuições de uma comunidade ativa, levando ao desenvolvimento de inúmeros pacotes que ampliam suas capacidades.
{{% /alert %}}

## Importância na Aprendizagem de Máquina

- **Auxilia no entendimento de outros modelos**: Compreender seus fundamentos ajuda a entender e implementar técnicas mais complexas;

- **Interpretável** Tem a vantagem da interpretabilidade dos coeficientes do modelo. O que pode oferecer _insights_ diretos sobre a influência de cada variável independente na variável dependente. Em problemas onde a explicabilidade do modelo é tão ou mais importante que a precisão isso é uma vantagem.

## Exemplos de Aplicação

- **Previsão de Preços de Imóveis:** A regressão pode ser usada para prever o preço de uma casa com base em características como localização, tamanho, e número de quartos. Esse é um dos exemplos mais comuns.
Esta abordagem permite que compradores e vendedores tenham uma melhor compreensão do valor de mercado de uma casa, considerando fatores relevantes que influenciam o preço.

- **Análise de Vendas:** Empresas utilizam regressão para prever vendas futuras com base em dados históricos, ajudando na tomada de decisões estratégicas. Essa previsão é utilizada na tomada de decisões estratégicas, como planejamento de estoque e campanhas de marketing. A capacidade de antecipar mudanças na demanda permite que as empresas se adaptem rapidamente ao mercado, melhorando sua eficiência operacional e maximizando lucros.

- **Ciências da Saúde:** Pesquisadores utilizam regressão para analisar a relação entre variáveis como idade, pressão arterial e colesterol, ajudando a identificar fatores de risco para doenças. Este tipo de análise ajuda a estabelecer correlações essenciais para o desenvolvimento de estratégias de prevenção e tratamento.

- **Engenharia:** No controle de qualidade, a regressão pode ajudar a prever a resistência de materiais com base em suas propriedades físicas e químicas. Essa aplicação visa garantir a segurança e eficácia dos materiais utilizados em construção e manufatura. Ao identificar as propriedades que afetam a resistência, engenheiros podem otimizar processos de produção e desenvolver materiais mais robustos.

## Vantagens e Limitações

### Vantagens

- **Simplicidade e Interpretação:** Fácil de implementar e interpretar, permitindo que usuários compreendam rapidamente as relações entre variáveis. Sendo acessível para iniciantes;

- **Eficiência Computacional:** Requer menos recursos computacionais em comparação com modelos mais complexos. Essa eficiência a torna ideal para análise de grandes conjuntos de dados, onde a velocidade e a economia de recursos são críticas.

- **Base para Modelos Complexos:** Serve como base para entender e desenvolver modelos de Machine Learning mais avançados. Ela oferece uma compreensão inicial dos dados, permitindo que pesquisadores e analistas desenvolvam modelos mais sofisticados, como regressão polinomial e redes neurais, a partir desse fundamento. Sendo essa uma etapa inicial relevante no processo de modelagem e análise de dados.

### Limitações

- **Linearidade:** Assume que a relação entre variáveis é linear, o
    que pode não ser verdade para todos os conjuntos de dados.

- **Sensibilidade a Outliers:** Outliers podem influenciar
    significativamente os resultados da regressão linear.

- **Assunção de Independência:** Pressupõe que as variáveis
    independentes são realmente independentes umas das outras, o que
    pode não ser o caso.

## Referências

<a id="1"></a>
Galton, F. (1886). Regression towards mediocrity in hereditary stature. The Journal of the Anthropological Institute of Great Britain and Ireland, 15, 246-263.

<a id="2"></a>
Pearson, K. (1896). VII. Mathematical contributions to the theory of evolution.—III. Regression, heredity, and panmixia. Philosophical Transactions of the Royal Society of London. Series A, containing papers of a mathematical or physical character, (187), 253-318.

<a id="3"></a>
Edwards, A. W. (2005). RA Fischer, statistical methods for research workers, (1925). In Landmark writings in western mathematics 1640-1940 (pp. 856-870). Elsevier Science.

<a id="4"></a>
Chambers, J. M. (1992). Linear models. In ‘Statistical models in S’.(Eds JM Chambers, TJ Hastie) pp. 95–144.

<a id="5"></a>
McKinney, W. (2010, June). Data structures for statistical computing in Python. In SciPy (Vol. 445, No. 1, pp. 51-56).