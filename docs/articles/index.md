# A Systematic Literature Review of Machine Learning Techniques for Software Effort Estimation  Models

## abstract

Estimating software projects is a challenging
but necessary process in software development. Predicting the
effort needed to build software is an essential part of the project
life cycle. This paper examines a variety of machine learning
algorithms for estimating effort. There has been a significant
increase in research on effort estimation with machine learning
approaches during the last two decades, with the objective of
improving
estimation
accuracy. To
forecast effort, the
estimation techniques such as expert judgment, COCOMO,
analogy based, putnam model, and machine learning are used.
The algorithmic models' low accuracy and
unreliable
architecture resulted in substantial software project risks. As a
result, it is essential to predict the cost of project on an annual
basis and compare it to alternative methods. However, the effort
prediction using machine learning is still limited because a single
technique cannot be treated as best. This paper's main goal is to
present a review of several machine learning approaches for
predicting effort.

:::info
estimar projetos de software é um processo desafiador, mas necessário no desenvolvimento de software. Prever o esforço necessário para construir software é uma parte essencial do ciclo de vida do projeto. Este artigo examina uma variedade de algoritmos de aprendizado de máquina para estimar esforço. Houve um aumento significativo na pesquisa sobre estimativa de esforço com abordagens de aprendizado de máquina durante as últimas duas décadas, com o objetivo de melhorar a precisão da estimativa. Para prever o esforço, são utilizadas técnicas de estimativa como julgamento de especialistas, COCOMO, baseado em analogia, modelo putnam e aprendizado de máquina. A baixa precisão e a arquitetura não confiável dos modelos algorítmicos resultaram em riscos substanciais no projeto de software. Como resultado, é essencial prever o custo do projeto anualmente e compará-lo com métodos alternativos. No entanto, a previsão de esforço utilizando aprendizado de máquina ainda é limitada porque uma única técnica não pode ser tratada como a melhor. O principal objetivo deste artigo é apresentar uma revisão de diversas abordagens de aprendizado de máquina para previsão de esforço.
:::

## Introdução

It is generally known that the software industry fails to deliver accurate cost, time, and effort estimates [1]. 
The success of software projects depends on accurate estimation.
The need for smarter, more inventive software is higher than ever, requiring the development of better, yet less expensive, and better software solutions. 
Estimation methods are used in process of project management to estimate cost, effort, and
time, which generally involves the development process.
Software project forcasting is exceedingly challenging due to the intangible nature of computer software and the inherent high margin of error in estimating [2]. 
Estimating the amount of effort needed to produce software in terms of money, time, and labour is known as effort estimation. 
The effort is measured in person-months [3]. 
For estimating effort, different models have been presented. 
Firstly, effort prediction was done using analogy-based estimation, expert judgment,
and use case point. 
Following that, existing machine learning approaches such as neural networks, random forests, linear
regression, decision trees, naive bayes, support vector machine, and so on are used to predict effort [4]. 
Due to its ability to learn from previously completed projects, machine learning approaches reliably estimate appropriate results.
These methods provide extremely acurate predictions.
Despite massive studies on software development effort estimation strategies, not a single strategy has been demonstrated to be better than others in all situations. 
There has recently been a rise in research on predicting software effort using machine learning approaches [5].

:::info
É geralmente conhecido que a indústria de software não consegue fornecer estimativas precisas de custo, tempo e esforço [1]. 
O sucesso dos projetos de software depende de estimativas precisas.
A necessidade de software mais inteligente e inventivo é maior do que nunca, exigindo o desenvolvimento de soluções de software melhores, porém menos dispendiosas e melhores. 
Métodos de estimativa são usados ​​no processo de gerenciamento de projetos para estimar custo, esforço e
tempo, que geralmente envolve o processo de desenvolvimento.
A previsão de projetos de software é extremamente desafiadora devido à natureza intangível do software de computador e à alta margem de erro inerente na estimativa [2]. 
Estimar a quantidade de esforço necessária para produzir software em termos de dinheiro, tempo e mão de obra é conhecida como estimativa de esforço. 
O esforço é medido em pessoas-mês [3]. 
Para estimar o esforço, diferentes modelos foram apresentados. 
Em primeiro lugar, a previsão do esforço foi feita utilizando estimativas baseadas em analogias, julgamento de especialistas,
e ponto de caso de uso. 
Depois disso, abordagens existentes de aprendizado de máquina, como redes neurais, florestas aleatórias, sistemas lineares
regressão, árvores de decisão, bayes ingênuos, máquina de vetores de suporte e assim por diante são usados ​​para prever o esforço [4]. 
Devido à sua capacidade de aprender com projetos concluídos anteriormente, as abordagens de aprendizado de máquina estimam com segurança os resultados apropriados.
Esses métodos fornecem previsões extremamente precisas.
Apesar de estudos massivos sobre estratégias de estimativa de esforço de desenvolvimento de software, nem uma única estratégia demonstrou ser melhor que outras em todas as situações. 
Recentemente, houve um aumento na pesquisa sobre a previsão do esforço de software usando abordagens de aprendizado de máquina [5].
:::