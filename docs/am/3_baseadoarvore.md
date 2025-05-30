# Modelos Baseados em Árvore

## Métodos simbólicos

Englobam modelos cujo objetivo é representar explicitamente, através de estruturas simbólicas\footnote{Neste contexto, o termo \textbf{símbolo} refere-se à abstração de conceitos, objetos ou relações do mundo real, que propriamente os representam ou a suas características e estados.

Por conseguinte, \textbf{estruturas simbólicas} são agrupamentos desses elementos fundamentais, de modo a representar conhecimentos e relacionamentos mais complexos.}, o conhecimento extraído do conjunto de dados. Esses métodos facilitam a interpretação do resultado por seres humanos, visto que asseguram ``{[}\ldots{]} uma compreensibilidade maior do processo decisório {[}\ldots{]}, estando mais alinhado{[}s{]} aos princípios de que os modelos de AM devem também ser `explicáveis' (\emph{Explainable Machine Learning}) para garantir maior transparência em sua operação.'' (FACELI et al., 2023, p.~78). Em contrapartida, se utilizados isoladamente, têm menor acurácia preditiva em comparação a outros modelos, ditos ``caixa-preta''\footnote{A literatura comumente	utiliza a expressão ``caixa-preta'' (\emph{black box}) para se referir a modelos cujo processo decisório não é facilmente inferível ou interpretável por seres humanos, como é o caso das redes neurais artificiais. A propósito, vide o fichamento \href{../neural-networks-and-learning-machines-simon-haykin/_introduction.md}{1.2}, que contém algumas referências à terminologia no âmbito das redes neurais, em especial as \href{../neural-networks-and-learning-machines-simon-haykin/_introduction.md\#notas}{notas} 1 e 19.}.

Não obstante, é importante destacar que, ``atualmente, existem algoritmos eficientes para indução de árvores de decisão ou conjuntos de regras e de aplicação eficiente, com um desempenho equivalente ao de outros modelos (como redes neurais e SVM), mas com maior grau de interpretabilidade. {[}\ldots{]} A combinação de múltiplos modelos de árvores em comitês (\emph{ensembles}) também tem se mostrado competitiva e é uma abordagem frequentemente empregada para aumentar o desempenho preditivo desses modelos.'' (FACELI et al., 2023, p.~97).

\subsection[6.1 Modelos baseados em árvores]{\texorpdfstring{6.1 Modelos baseados em árvores\footnote{Ver o \href{../../suplementos/03-estruturas-de-dados.md}{suplemento 3}, sobre estruturas de dados, para informações adicionais sobre \textbf{árvores}.}}{6.1 Modelos baseados em árvores}}\label{modelos-baseados-em-uxe1rvores3}

Os modelos baseados em árvores utilizam a estrutura de dados homônima para solucionar problemas de classificação ou regressão, casos em que os algoritmos são respectivamente denominados \textbf{árvores de decisão} ou \textbf{árvores de regressão}. Em ambos os casos, a forma de se interpretar o modelo e de construir o algoritmo indutor da própria árvore são bastante similares e, de modo geral, o problema é abordado \textbf{recursivamente} por meio da estratégia da \textbf{divisão e conquista}\footnote{A \textb{recursividade} é uma característica de determinados algoritmos de se chamarem a si mesmos, uma ou mais vezes, a fim de fracionar um problema em tantos problemas menores quantos forem necessários, até que seja possível resolver o problema original.

Normalmente, isso é feito por meio da abordagem da \textbf{divisão e conquista}, que em cada nível de recursão aplica três etapas: divisão, conquista e combinação (Cormen et al., 2012).} sem \emph{backtracking}.


>``Usualmente, os algoritmos exploram heurísticas que localmente executam uma pesquisa olha para a frente um passo. Uma vez que uma decisão é tomada, ela nunca é reconsiderada. Essa pesquisa de subida de encosta (\emph{hill-climbing}) sem \emph{backtracking} é suscetível aos riscos usuais de convergência de uma solução ótima localmente que não é ótima globalmente. Por outro lado, essa estratégia permite construir árvores de decisão em tempo linear no número de exemplos.'' (FACELI et al., 2023, p.~80).

Nesse sentido, ``um problema complexo é dividido em problemas mais
simples, aos quais recursivamente é aplicada a mesma estratégia. As
soluções dos subproblemas podem ser combinadas, na forma de uma árvore,
para produzir uma solução do problema complexo. A força dessa proposta
vem da capacidade de dividir o espaço de instâncias em subespaços e cada
subespaço é ajustado usando diferentes modelos.'' (FACELI et al., 2023,
p.~78).

``Formalmente, uma árvore de decisão\footnote{No livro, os autores
	utilizam o termo árvore de decisão para se referir, indistintamente,
	às árvores de decisão ou de regressão, inclusive neste caso, dado que
	a interpretação dos modelos e a indução da árvore são bastante
	similares. Todavia, ressalvam que, se necessária, haverá a devida
	distinção.} é um grafo direcionado acíclico em que cada nó ou é um nó
de divisão, com dois ou mais sucessores, ou um nó folha.'' (FACELI et
al., 2023, p.~79). Os \textbf{nós de divisão} possuem testes
condicionais de acordo com o valor do atributo que representam; os
\textbf{nós folha} são funções que representam as saídas do modelo,
possuindo os valores da variável alvo. Cada nó da árvore corresponde a
uma região no espaço definido pelos atributos.

> ``As regiões definidas pelas folhas da árvore são mutuamente excludentes, e a reunião dessas regiões cobre todo o espaço definido pelos atributos. A interseção das regiões abrangidas por quaisquer duas folhas é vazia. A união de todas as regiões (todas as folhas) é U. Uma árvore de decisão abrange todo o espaço de instâncias. Esse fato implica que uma árvore de decisão pode fazer predições para qualquer exemplo de entrada. {[}\ldots{]} As condições ao longo de um ramo (um percurso entre a raiz e uma folha) são conjunções de condições e os ramos individuais são disjunções. Assim, cada ramo forma uma regra com uma parte condicional e uma conclusão. A parte condicional é uma conjunção de condições. Condições são testes que envolvem um atributo particular, operador {[}\ldots{]} e um valor do domínio do atributo.'' (FACELI et al., 2023, p.~79).


%Para exemplificar, vejamos a imagem a seguir:
%\pandocbounded{\includegraphics[keepaspectratio]{../../imagens/21_am_faceli_arvore_de_decisao.png}}
Figura 21 --- Árvore de decisão e regiões de decisão no espaço de
objetos (FACELI et al., 2023, p.~79).

Os modelos baseados em árvores possuem \textbf{vantagens} como
\textbf{flexibilidade} --- por serem não paramétricos, não pressupõem
alguma distribuição específica de dados\footnote{``O espaço de objetos é
	dividido em subespaços e cada subespaço é ajustado com diferentes
	modelos. Uma árvore de decisão fornece uma cobertura exaustiva do
	espaço de instâncias.'' (FACELI et al., 2023, p.~89).} ---,
\textbf{robustez} --- lidam bem com transformações de variáveis que
preservem a ordem dos dados, não modificando a estrutura da árvore e a
lógica do processo decisório\footnote{``Árvores univariáveis são
	invariantes a transformações (estritamente) monótonas de variáveis de
	entrada. {[}\ldots{]} Como consequência dessa invariância, a
	sensibilidade a distribuições com grande cauda e \emph{outliers} é
	também reduzida (Friedman, 1999).'' (FACELI et al., 2023, p.~89).}
---, \textbf{autonomia na seleção de atributos}\footnote{``O processo de
	construção de uma árvore de decisão seleciona os atributos a usar no
	modelo de decisão. Essa seleção de atributos produz modelos que tendem
	a ser bastante robustos contra a adição de atributos irrelevantes e
	redundantes.'' (FACELI et al., 2023, p.~89).},
\textbf{interpretabilidade}\footnote{``Decisões complexas e globais
	podem ser aproximadas por uma série de decisões mais simples e locais.
	Todas as decisões são baseadas nos valores dos atributos usados para
	descrever o problema.'' (FACELI et al., 2023, p.~89).} e
\textbf{eficiência}\footnote{``O algoritmo para aprendizado de árvore de
	decisão é um algoritmo guloso que é construído de cima para baixo
	(\emph{top-down}), usando uma estratégia dividir para conquistar sem
	\emph{backtracking}. Sua complexidade de tempo é linear com o número
	de exemplos.'' (FACELI et al., 2023, p.~89).}. Dentre as
\textbf{desvantagens}, estão a \textbf{replicação}\footnote{``O termo
	refere-se à duplicação de uma sequência de testes em diferentes ramos
	de uma árvore de decisão, levando a uma representação não concisa, que
	também tende a ter baixa acurária preditiva {[}\ldots{]}.'' (FACELI et
	al., 2023, p.~89).}, a \textbf{instabilidade}\footnote{``Pequenas
	variações no conjunto de treinamento podem produzir grandes variações
	na árvore final {[}\ldots{]}. A estratégia da partição recursiva
	implica que a cada divisão que é feita o dado é dividido com base no
	atributo de teste. Depois de algumas divisões, há usualmente muito
	poucos dados nos quais a decisão se baseia. Há uma forte tendência a
	inferências feitas próximo das folhas serem menos confiáveis que
	aquelas feitas próximas da raiz.'' (FACELI et al., 2023, p.~90).} e a
ineficiência em cenários específicos, como dados com \textbf{valores
	ausentes}\footnote{``Uma árvore de decisão é uma hierarquia de teses. Se
	o valor de um atributo é desconhecido, isso causa problemas em decidir
	que ramo seguir.'' (FACELI et al., 2023, p.~89).} e \textbf{atributos
	contínuos}\footnote{``Nesse caso, uma operação de ordenação é solicitada
	para cada atributo contínuo de cada nó de decisão.'' (FACELI et al.,
	2023, p.~89).}.

Há também modelos não convencionais como as \textbf{árvores de modelos e
	de opções}, alternativas que podem ser usadas em tarefas de regressão e
classificação, respectivamente.


> ``Uma \textbf{árvore de modelos} (do inglês \emph{model tree}) {[}\ldots{]} combina árvore de regressão com equações de regressão. Esse tipo de árvore funciona da mesma maneira que uma árvore de regressão, porém \textbf{os nós folha contêm expressões lineares em vez de valores agregados (médias ou medianas).} A estrutura da árvore divide o espaço dos atributos em subespaços, e os exemplos em cada um dos subespaços são aproximados por uma função linear. A \emph{model tree} é \textbf{menor e mais compreensível} que uma árvore de regressão e, mesmo assim, apresenta um erro médio menor na predição.'' (FACELI et al., 2023, p.~96, destaquei).

Em tarefas de classificação, as \textbf{árvores de opção} ``{[}\ldots{]}
podem incluir \emph{nós de opção}, que trocam o usual teste no valor de
um atributo por um conjunto de testes, cada um dos quais sobre o valor
de um atributo. Um nó de opção é como um nó \emph{ou} em árvores
\emph{e-ou}. \textbf{Na construção da árvore, em vez de selecionar o
	\emph{melhor} atributo, são selecionados todos os atributos promissores,
	aqueles com maior valor do ganho de informação.} Para cada atributo
selecionado, uma árvore de decisão é construída. É de salientar que uma
árvore de opção pode ter três tipos de nós: nós com somente um atributo
teste - \emph{nós de decisão}; nós com disjunções dos atributos de teste
- \emph{nós de opção}; e nós folhas.'' (FACELI et al., 2023, p.~97,
destaquei). Em contrapartida, o consumo de recursos computacionais,
notadamente memória e espaço para armazenamento, é substancialmente
maior.

\subsubsection{6.1.1 Regras de divisão em tarefas de
	classificação}\label{regras-de-divisuxe3o-em-tarefas-de-classificauxe7uxe3o}

As regras de divisão servem para \textbf{atenuar a impureza} dos
conjuntos de dados e balizam a construção da árvore de decisão no
intuito de \textbf{maximizar a homogeneidade} dos subconjuntos gerados
em cada recursão, de modo que garantir a congruência dos atributos no
tocante à seleção daqueles que melhor discriminam cada classe
(\emph{goodness of split}).

``Uma proposta natural é rotular cada subconjunto da divisão por sua
classe mais frequente e escolher a divisão que tem menores erros''
(FACELI et al., 2023, p.~81), e as diversas propostas para isso
convergem para a conclusão de que ``{[}\ldots{]} uma divisão que mantém
a proporção de classes em todo o subconjunto não tem utilidade, e uma
divisão na qual cada subconjunto contém somente exemplos de uma classe
tem utilidade máxima.'' (FACELI et al., 2023, p.~81).

Assim, a melhor divisão é aquela que minimiza a impureza e,
consequentemente, heterogeneidade dos subconjuntos. Em sentido
logicamente contrário, portanto, é de se concluir que a divisão ideal
busca maximizar a homogeneidade dos dados em cada subconjunto.

Conforme Martin (1997), os autores distinguem as métricas para avaliar a
qualidade das divisões em subconjuntos em três grandes grupos
(\textbf{funções de mérito}), considerando diferentes critérios:
\textbf{(1)} baseadas na diferença entre a \textbf{distribuição de dados
	antes e depois da divisão}, que enfatizam a \textbf{pureza dos
	subconjuntos}; \textbf{(2)} baseadas em \textbf{diferenças entre os
	subconjuntos}, cujo enfoque é a \textbf{disparidade entre os
	subconjuntos} após a divisão; e \textbf{(3)} baseadas na
\textbf{confiabilidade dos subconjuntos}, isto é, em medidas
estatísticas de independência capazes de denotar que cada nó/subconjunto
(atributo) é efetivamente adequado para produzir boas previsões, em que
a ênfase é no \textbf{peso da evidência}.

Embora não haja consenso em relação à superioridade, a escolha por algum
critério parece ser superior à divisão aleatória de atributos.

Nas tarefas de classificação, as regras baseadas no \textbf{ganho de
	informação} e no \textbf{índice de Gini} são as mais comuns.

\paragraph{6.1.1.1 Baseadas no ganho de
	informação}\label{baseadas-no-ganho-de-informauxe7uxe3o}

Esta regra é baseada no conceito de \textbf{entropia}, que informa a
\textbf{aleatoriedade de uma variável}, isto é, \textbf{quantifica a
	dificuldade} de predizê-la. A entropia também pode ser compreendida como
uma medida da desordem ou impureza do conjunto de dados, e é medida em
logaritmos na base 2. Logo, quanto maior a entropia, mais difícil será
predizer o valor dessa variável aleatória, e \textbf{a árvore de decisão
	é construída de modo a minimizar a dificuldade} de predizer a variável
alvo.

>	``A cada nó de decisão, o atributo que mais reduz a aleatoriedade da variável alvo será escolhido para dividir os dados. {[}\ldots{]} Os valores de um atributo definem partições {[}leia-se divisões{]} no conjunto de exemplos. Para cada atributo, o ganho de informação mede a redução na entropia nas partições obtidas de acordo com os valores do atributo. Informalmente, o ganho de informação é dado pela diferença entre a entropia do conjunto de exemplos e a soma ponderada da entropia das partições. A construção da árvore de decisão é guiada pelo objetivo de reduzir a entropia, isto é, a aleatoriedade (dificuldade para predizer) da variável alvo.'' (FACELI et al., 2023, p.~81). 

Em cada nó de divisão da árvore, o atributo que mais reduzir a entropia, consequentemente maximizando o ganho de informação, será escolhido para resolver o subproblema --- leia-se, a divisão em subconjunto naquela etapa recursiva. Logo, é esperado que a cada divisão ocorra a diminuição da aleatoriedade --- incerteza em relação à classificação correta da variável alvo --- em virtude da consistência dos atributos que conformam o subconjunto como sendo aqueles que melhor discriminam as classes.

Por isso é que, essencialmente, o ganho de informação consiste na redução da aleatoriedade resultante da diferença entre a entropia de todo o conjunto de dados e a dos subconjuntos.

\paragraph[6.1.1.2 Baseadas no índice de Gini ]{\texorpdfstring{6.1.1.2
		Baseadas no índice de Gini
		\footnote{Na Economia, o \textbf{índice de Gini} é uma \textbf{medida de
				desigualdade} muito empregada para analisar a distribuição de renda,
			mas que pode ser usada para medir o grau de desigualdade de qualquer
			distribuição estatística, definido como a razão entre a \textbf{área
				de desigualdade}, obtida entre a \textbf{linha da perfeita igualdade}
			e a \textbf{curva de Lorenz}, e a área do triângulo formado pelos
			eixos do gráfico e a linha de perfeita igualdade (Hoffmann, 2006).
			Analogicamente, no contexto da aprendizagem de máquina, é possível
			trasladar esse raciocínio para a desigualdade --- leia-se
			heterogeneidade --- da distribuição dos elementos em um conjunto ou
			subconjunto de dados.}}{6.1.1.2 Baseadas no índice de Gini }}\label{baseadas-no-uxedndice-de-gini-15}

É uma métrica de \textbf{impureza} dos nós de decisão (subconjuntos).
\textbf{Avalia a probabilidade de que exemplos escolhidos ao acaso
	pertençam a classes diferentes, mas estejam no mesmo subconjunto.}
Quanto menor o índice de Gini, mais homogêneo --- e menos impuro --- é o
subconjunto. Nesse sentido, o atributo que melhor discrimina a classe é
aquele que minimiza o índice de Gini e, via de consequência, reduz a
impureza e aumenta a homogeneidade dos subconjuntos.

\subsubsection{6.1.2 Regras de divisão em tarefas de
	regressão}\label{regras-de-divisuxe3o-em-tarefas-de-regressuxe3o}

Nas tarefas de regressão, o critério mais usual é calcular a
\textbf{média do erro quadrático} (erro quadrático médio --- EQM ou
\textbf{\emph{mean squared error}} --- MSE), objetivando que dessa
divisão resultem subconjuntos compostos por elementos cujos valores
aproximem-se entre si, consequentemente minimizando o erro. ``Por esse
motivo, a constante associada às folhas de uma árvore de regressão é a
média dos valores do atributo alvo dos exemplos de treinamento que caem
na folha.'' (FACELI et al., 2023, p.~85).

Uma forma de avaliar a qualidade das divisões foi proposta por Breiman
et al.~(1984) e consiste na \textbf{redução do desvio padrão}
(\textbf{\emph{standard deviation reduction}} - SDR), a ser calculada
para cada subconjunto possível, de modo que seja o que ensejar a menor
variância.

\subsubsection{6.1.3 Valores desconhecidos}\label{valores-desconhecidos}

Submeter valores desconhecidos ou indeterminados ao modelo, isto é, que
não foram explicitamente contemplados dentre os dados de treinamento,
pode levar a resultados indesejados, ``uma vez que uma árvore de decisão
constitui uma hierarquia de testes {[}\ldots{]}'' (FACELI et al., 2023,
p.~86).

Para resolver o chamado \textbf{problema do valor desconhecido}, há na
literatura diversas propostas, como \textbf{(1)} atribuir-lhe o valor
mais frequente; \textbf{(2)} considerá-lo também um valor possível para
o atributo em questão; \textbf{(3)} associar probabilidades a todos os
possíveis valores do atributo (utilizada no algoritmo C4.5); ou
\textbf{(4)} implementar a estratégia da divisão substituta, que
preconiza a utilização de atributos que gerem divisões similares àquela
obtida pelo atributo selecionado para criar uma lista alternativa de
atributos, possibilitando a busca por outro, que corresponda ao valor
desconhecido (utilizada no algoritmo CART).

\subsubsection{6.1.4 Estratégias de poda}\label{estratuxe9gias-de-poda}

A poda de uma árvore de decisão (\emph{decision tree pruning}) consiste
na diminuição de seu tamanho, substituindo por folhas os nós
demasiadamente profundos\footnote{O caminho mais longo entre as
	extremidades determina a \textbf{altura} da árvore, enquanto a
	\textbf{profundidade} é a quantidade de nós ou camadas horizontais
	existentes nesse caminho (Brookshear, 2013).} (eliminação de ramos ou
subárvores), com o objetivo de aumentar a \textbf{confiabilidade} do
modelo e tornar o processo decisório ainda mais \textbf{compreensível}.
O procedimento tende a melhorar a capacidade de \textbf{generalização}
do modelo e é de especial importância em cenários com dados ruidosos.

> ``Dados ruidosos levantam dois problemas. O primeiro é que as árvores	induzidas classificam novos objetos em um modo não confiável. Estatísticas calculadas nos nós mais profundos de uma árvore têm baixos níveis de importância em função do pequeno número de exemplos que chegam nesses nós. Nós mais profundos refletem mais o conjunto de treinamento (superajuste) e aumentam o erro em razão da variância do classificador. O segundo é que a árvore induzida tende a ser grande e, portanto, difícil para compreender.'' (FACELI et al., 2023, p.~86).

É possível realizar a poda concomitantemente à construção da árvore,
interrompendo o processo conforme algum critério predeterminado, e
outras que aguardam o término, respectivamente denominadas
\textbf{pré-poda} e \textbf{pós-poda}. Não obstante, ``tal como as
regras de divisão, a poda é um domínio em que nenhuma proposta existente
é a melhor para todos os casos. A poda é um enviesamento em direção à
simplicidade. Se o domínio do problema admite soluções simples, então a
poda é uma opção eficiente (Schaffer, 1993).'' (FACELI et al., 2023,
p.~88).

A \textbf{pré-poda} implementa regras que interrompem a construção de
ramos que aparentemente não contribuiriam para incrementar a acurácia
preditiva da árvore, evitando desde o início a criação de nós
considerados inúteis, o que economiza tempo e recursos computacionais.
Embora a literatura enumere diversas regras possíveis, são
majoritariamente aceitas a assunção de que \textbf{(1)} todos os objetos
que alcancem um determinado nó são da mesma classe e/ou de que
\textbf{(2)} todos os elementos que o alcancem possuem características
idênticas, embora não necessariamente pertençam à mesma classe.

Todavia, as estratégias de \textbf{pós-poda} são mais comuns e resultam
em modelos mais confiáveis, embora o processo construtivo seja mais
demorado porque ``uma árvore completa, superajustada aos dados de
treinamento, é gerada e podada posteriormente.'' (FACELI et al., 2023,
p.~87). Logicamente que, por esse motivo, há maior consumo de tempo e
recursos computacionais. Dentre os métodos de pós-poda, são elencados os
\textbf{(1)} baseados nas medidas de \textbf{erro estático e erro de
	\emph{backed-up}}; \textbf{(2)} a poda \textbf{custo de complexidade},
um dos mais utilizados, introduzido por Breiman et al.~(1984) no
algoritmo CART; e \textbf{(3)} a poda \textbf{pessimista}, apresentada e
adotada por Quinlan (1988) no algoritmo C4.5\footnote{Especialmente os
	métodos do custo de complexidade e da poda pessimista são abordados em
	detalhe na seção 6.3.2 do livro (p.~87/88).}.