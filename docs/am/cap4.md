---
title: Modelos baseados em regras
---

# Modelos baseados em regras

Uma regra de decisão compara logicamente um atributo e os valores
conhecidos do domínio e, assim como nas árvores de decisão, seu espaço
de hipóteses é dado sob a forma disjuntiva.

A despeito da grande semelhança, as regras de decisão flexibilizam
algumas características inerentes às árvores de decisão que podem ser
desvantajosas, como a replicação e a instabilidade, e tornam o processo
decisório modular, visto que as regras podem ser avaliadas isoladamente,
sem que modificações realizadas em determinada regra condicional afetem
as subsequentes.

> ``Regras de decisão e árvores de decisão são bastante similares em suas
	formas de representação para expressar generalizações dos exemplos.
	Ambas definem superfícies de decisão similares. {[}\ldots{]}
	\textbf{Como as árvores de decisão cobrem todo o espaço de instâncias, a
		vantagem é que qualquer exemplo é classificado por uma árvore de
		decisão. Entretanto, cada teste em um nó tem um contexto definido por
		testes anteriores, definidos nos nós no caminho, que podem ser
		problemáticos se levarmos em conta a interpretabilidade. Por outro lado,
		as regras são modulares, ou seja, podem ser interpretadas isoladamente.}
	Cada regra cobre uma região específica do espaço de instâncias. A união
	de todas as regras pode ser menor que o Universo.'' (FACELI et al.,
	2023, p.~90, destaquei).

%Para exemplificar, vejamos a imagem a seguir:
%\pandocbounded{\includegraphics[keepaspectratio]{../../imagens/22_am_faceli_regras_de_decisao.png}}
%Figura 22 --- Exemplos de superfícies de decisão desenhadas por um
%conjunto de regras (FACELI et al., 2023, p.~90).

A similaridade permite a conversão de árvores em conjuntos ou listas de
regras de decisão, tal que cada folha da árvore corresponda a uma regra.
Embora a regra contemple o percurso por toda a altura da árvore --- da
raiz à folha ---, é possível otimizar a representação, simplificando-a
por meio da remoção de condições redundantes ou irrelevantes.
Justifica-se essa abordagem:


>	``\textbf{Árvores de decisão extensas são de difícil compreensão porque
		o teste de decisão em cada nó aparece em um contexto específico,
		definido pelo resultado de todos os testes nos nós antecedentes.} O
	trabalho desenvolvido por Rivest (1987) apresenta as \emph{listas de
		decisão}, uma nova representação para a generalização de exemplos que
	estende as árvores de decisão. A grande vantagem dessa representação é a
	modularidade do modelo de decisão e, consequentemente, a sua
	interpretabilidade: cada regra é independente das outras regras, e pode
	ser interpretada isoladamente das outras regras. Como consequência,
	\textbf{a representação utilizando regras de decisão permite eliminar um
		teste em uma regra, mas reter o teste em outra regra.} Além disso, como
	a conjunção de condições é comutativa, \textbf{a distinção entre testes
		perto da raiz e testes perto das folhas desaparece.}'' (FACELI et al.,
	2023, p.~91, destaquei).

Nesse sentido, o \textbf{algoritmo de cobertura} é capaz de aprender
regras de decisão baseadas em exemplos e a qualidade das regras pode ser
avaliada pela quantidade de casos cobertos, tenham ou não sido
corretamente classificados.

>	``O algoritmo da cobertura define o processo de aprendizado como um
	processo de procura: dados um conjunto de exemplos classificados e uma
	linguagem para representar generalizações dos exemplos, o algoritmo
	procede, para cada classe, a uma procura heurística. Tipicamente, o
	algoritmo procura regras da forma: se \(Atributo_i = Valor_j\) e
	\(Atributo_l = Valor_k\) \ldots{} então \(Classe_z\). A procura pode
	proceder a partir da regra mais geral, ou seja, uma regra sem parte
	condicional, para regras mais específicas, acrescentando condições
	\textbf{{[}busca \emph{top-down} orientada pelo modelo{]}}; ou a partir
	de regras muito específicas {[}\ldots{]} para regras mais gerais,
	eliminando restrições \textbf{{[}busca \emph{bottom-up} orientada pelos
		dados{]}}. O processo de procura é guiado por uma função de avaliação
	das hipóteses. Essa função estima a qualidade das regras que são geradas
	durante o processo. {[}\ldots{]} \textbf{Dado um conjunto de exemplos de
		classes diferentes, o \emph{algoritmo de cobertura} consiste em aprender
		uma regra para uma das classes, removendo o conjunto de exemplos
		cobertos pela regra (ou o conjunto de exemplos positivos)}, e repetir o
	processo. O processo termina quando só há exemplos de uma única
	classe.'' (FACELI et al., 2023, p.~91/92, destaquei).

Pode haver conflito entre duas ou mais regras, caso em que será
necessário estabelecer algum critério de escolha. É importante observar
que, diferentemente dos métodos \emph{bottom-up}, os \emph{top-down}
induzem conjuntos ordenados de regras. Portanto, nestes, a execução do
algoritmo é interrompida diante da primeira regra que satisfaça à
condição de parada, enquanto naqueles, todas as regras aplicáveis são
testadas e, normalmente, o resultado será ponderado pela qualidade de
cada uma. Por esse motivo, é comum que algoritmos orientados por
processos \emph{top-down} contenham uma regra que específica para a
classificação de exemplos desconhecidos.
