---
title: Redes neurais artificiais
---

# Redes neurais artificiais

Desde o início, o estudo das redes neurais artificiais foi pautado pela observação de que tanto o cérebro humano quanto os computadores convencionais são sistemas de processamento de informações e, por conseguinte, realizam trabalho computacional. Não apenas seu funcionamento é bastante distinto, mas a capacidade computacional do cérebro - não só humano, mas também o de outros animais - supera, em muito, a dos computadores digitais. 

No entanto, em breve a capacidade computacional dos computadores digitais irá superar a capacidade do cérebro humano. O projeto DeepSouth, desenvolvido por estudantes da universidade de \textit{Western Sydney University}, será o primeiro computador do mundo a simular todas as 228 trilhões sinapses do cérebro \cite{Deepsouth}. Seu lançamento está planejado para meados de 2024.

O cérebro é um sistema de processamento de informações complexo, não linear e paralelo. Suas estruturas fundamentais são os neurônios, organizados de modo a realizar tarefas computacionais, a exemplo do reconhecimento de padrões, \textcolor{blue}{tais como, reconhecimento de formas e rostos}. Portanto, assim no cérebro como nas redes artificiais, os neurônios constituem as unidades de processamento da informação.

% Em linhas gerais, pode-se dizer que uma rede neural artificial é um modelo computacional inspirado no modo como o cérebro realiza o processamento de informações. Nas palavras de Haykin (2009, p. 2), "[...] a neural network is a machine that is designed to model the way in which the brain performs a particular task or function of interest".

\begin{tcolorbox}
	Os neurônios constituem as unidades de processamento da informação, tanto no cérebro quanto em redes neurais!
\end{tcolorbox}

A plasticidade cerebral permite ao ser humano criar conexões para execução de funções não planejadas originalmente. Nascemos com a habilidade de reconhecer padrões e desenvolvemos naturalmente essa capacidade, dado que existem na natureza vários padrões para esse aprendizado. Porém, sem essa habilidade da plasticidade seria impossível reconhecer as letras, formar palavras e consequentemente extrair conceitos. Pois não existem letras na natureza. Essa habilidade original de reconhecê padrões é adaptada para essa nova função de leitura caso o ser humano participe de um processo de aprendizado que acontece em nossa cultura. Nas redes neurais artificiais essa plasticidade, característica que permite a adaptação do indivíduo ao ambiente em que está inserido, é crucial para o aprendizado artificial. 

Em linhas gerais, pode-se dizer que uma rede neural artificial é um modelo computacional inspirado no modo como o cérebro realiza o processamento de informações. Nas palavras de Haykin (2009, p. 2), \textit{"[...] a neural network is a machine that is designed to model the way in which the brain performs a particular task or function of interest"} \cite{haykin2009neural}. Essa inspiração e interdisciplinaridade acontece em diversas outras áreas do conhecimento como por exemplo, os algoritmos genéticos inspirados na    reprodução e sobrevivência dos mais aptos \cite{de1999introduccao},  o método probabilístico que constrói soluções por meio da inteligência coletiva baseados em colônias de formigas \cite{de2007aplicaccoes}, e o  algoritmo baseado em amebas que conseguiu obter uma solução aproximada para o problema do caixeiro-viajante \cite{ameba2021}.

Já como definição formal, Haykin (2009) dá às redes neurais, vistas como uma máquina adaptativa, o seguinte conceito:

\begin{quote}
	``A neural network is a massively parallel distributed processor made up of simple processing units that has a natural propensity for storing experiential knowledge and making it available for use. It resembles the brain in two respects: 1. Knowledge is acquired by the network from its environment through a learning process. 2. Interneuron connection strengths, known as synaptic weights, are used to store the acquired knowledge.'' \cite{haykin2009neural}
\end{quote}

\begin{quote}
	"Uma rede neural é um processador maciçamente paralelamente distribuído constituído de unidades de processamento simples, que tem uma propensão natural para armazenar conhecimento experimental e torná-lo disponível para uso. Ela se assemelha ao cérebro em dois aspectos: 1. O conhecimento é adquirido pela rede a partir de seu ambiente através de um processo de aprendizagem. 2. Forças de conexão entre os neurônios, conhecidas como pesos sinápticos, são utilizadas para armazenar o conhecimento adquirido." \cite{haykin2001redes} 
\end{quote}

Assim, na medida do possível, as redes neurais artificiais assemelham-se ao cérebro humano, no sentido de serem interconexões de unidades computacionais mais simples.

O processo de aprendizagem (representado por um algoritmo de aprendiazagem) tradicionalmente aplicado em redes neurais consiste na modificação dos pesos sinápticos das conexões neuronais e deve resultar no alcance do objetivo almejado. Conforme \cite{haykin2009neural}, a técnica de modificação dos pesos sinápticos guarda muita similaridade com a teoria dos filtros adaptativos lineares (em inglês \textit{linear adaptive filter theory}). Cabe ressaltar também que a rede neural pode alterar sua própria estrutura (também chamado de topologia).

\subsection{Benefícios das redes neurais}

A capacidade computacional das redes neurais é resultado de sua estrutura massiva e paralelamente distribuída, bem como da habilidade de aprender e generalizar. 

\begin{quote}
	``A generalização se refere ao fato de a rede neural produzir saídas adequadas para entradas que não estavam presentes durante o treinamento (aprendizagem). Estas duas capacidades de processamento de informação [aprender e generalizar] tornam possível para as redes neurais resolver problemas complexos (de grande escala) que são atualmente intratáveis.''  \cite{haykin2001redes}
\end{quote}

Os métodos inspirados em modelos biológico, que buscam não apenas \textbf{simular o funcionamento} do sistema nervoso humano e o modo como o cérebro \textbf{adquire novos conhecimentos} --- ou seja, seu \textbf{processo de aprendizado} ---, mas alcançar \textbf{capacidade de processamento} semelhante e obter \textbf{máquinas inteligentes ou que se comportem de maneira aparentemente inteligente.} 

Assim como o cérebro é composto por uma grande quantidade de neurônios interconectados, perfazendo redes neurais que funcionam em paralelo e trocam informações através de sinapses.
 
Os principais componentes de um neurônio são:  dendritos, corpo celular e axônio. Os dendritos são prolongamentos dos neurônio especializados na recepção de estímulos   nervosos provenientes de outros neurônios ou do ambiente. Esses estímulos são então transmitidos para o corpo celular ou soma. O soma coleta as informações recebidas dos dendritos, as combina e processa. 
  
De acordo com a intensidade e frequência dos estímulos recebidos, o   corpo celular gera um novo impulso, que é enviado para o axônio. O axônio é um prolongamento dos neurônios, responsável pela condução dos   impulsos elétricos produzidos no corpo celular até outro local mais distante. O contato entre a terminação de um axônio e o  dendrito de outro neurônio é denominado sinapse. As sinapses são, portanto, as unidades que medeiam as interações entre os neurônios e podem ser excitatórias ou inibitórias \cite{faceli2023aprendizado}. 

As redes neurais artificiais (RNAs) são  formadas por unidades que implementam funções matemáticas a fim de simular a atividade neuronal e, de modo geral, \textbf{abstraem a compreensão da fisiologia do cérebro e dos processos biológicos de aprendizagem.}

\begin{quote}
``A procura por modelos computacionais ou matemáticos do sistema nervoso teve início na mesma época em que foram desenvolvidos os primeiros computadores eletrônicos, na década de 1940. Os estudos pioneiros na área foram realizados por McCulloch e Pitts (1943)

%\footnote{No artigo intitulado ``\emph{A Logical Calculus of Ideas Immanent in Nervous   Activity}'', publicado em 1943 no \emph{Bulletin of Mathematical Biophysics}, Warren McCulloch e Walter Pitts, ainda que simplificadamente, propuseram o primeiro modelo computacional/matemático de um neurônio biológico. À luz do conhecimento científico que se tinha àquele momento sobre a estrutura e o funcionamento das células nervosas, o trabalho foi norteado pelo estabelecimento das seguintes premissas: \textbf{(1) a atividade do neurônio é um processo ``tudo ou nada''} (\emph{``all-or-none'' process}); \textbf{(2) a ativação do neurônio exige que impulsos elétricos sejam recebidos em um período (latência) por uma determinada quantidade de sinapses}, que não se altera em razão da atividade prévia e/ou do estado da célula; \textbf{(3) o atraso sináptico foi o único intervalo de tempo considerado para induzir o modelo}, desprezando-se outros inerentes à geração e/ou transmissão de impulsos pelo sistema nervoso, pois irrelevantes; \textbf{(4) sinapses inibitórias impedem a ativação do neurônio naquele instante} (período refratário); e \textbf{(5) a estrutura da rede neural é invariante ao tempo}, assim como seus componentes. Noutras palavras, partiram da  assunção de que o processo computacional é binário, admitindo apenas dois estados não concomitantes --- ativo/ligado e inativo/desligado ---, mas invariante ao mero decurso do tempo na medida em que mudam de estado tão somente se verificada determinada condição, de modo que a qualquer instante o limiar de ativação, que é determinado pelo próprio neurônio e não por quaisquer características inerentes ao estímulo recebido, deve ser excedido em uma janela de latência para que o neurônio seja ativado. Os autores demonstraram que seu modelo era capaz de representar quaisquer proposições elementares e que, implementados em rede, conseguiriam calcular lógicas complexas. Não obstante, observaram limitações diante de disjunções exclusivas e que a incapacidade de modificar, extinguir ou criar sinapses impedia o processo de aprendizagem. Como se vê, o modelo constitui uma abstração da verdadeira fisiologia do sistema nervoso biológico.}. 

Em 1943, eles propuseram um modelo matemático de neurônio artificial, a unidade lógica com limiar (LTU, do inglês \emph{Logic Threshold Unit}), que podia executar funções lógicas simples. McCulloch e Pitts mostraram que a combinação de vários neurônios artificiais em sistemas neurais tem um elevado poder computacional, pois pode implementar qualquer função obtida pela combinação de funções lógicas. Entretanto, redes de LTUs não possuíam capacidade de aprendizado. {[}\ldots{]} Na década de 1970, houve um resfriamento das pesquisas em RNAs, principalmente com a {[}\ldots{]} limitação da rede Perceptron a problemas linearmente separáveis. Na década de 1980, o aumento da capacidade de processamento, as pesquisas em processamento paralelo e, principalmente, a proposta de novas arquiteturas de RNAs com maior capacidade de representação e de algoritmos de aprendizado mais sofisticados levaram ao ressurgimento da área.'' \cite{faceli2023aprendizado}.
\end{quote}

É possível definir as redes neurais artificiais como

\begin{quote}
``{[}\ldots{]} sistemas computacionais distribuídos compostos de unidades de processamento simples, densamente interconectadas
{[}\ldots{]}, conhecidas como neurônios artificiais, {[}que{]} computam funções matemáticas {[}\ldots{]}. As unidades {[}neurônios
artificiais{]} são dispostas em uma ou mais camadas e interligadas por um grande número de conexões, geralmente unidirecionais. Na maioria das arquiteturas, essas conexões, que simulam as sinapses biológicas, possuem pesos associados, que ponderam a entrada recebida por cada neurônio da rede {[}\ldots{]} {[}e{]} podem assumir valores positivos ou negativos, dependendo de o comportamento da conexão ser excitatório ou inibitório, respectivamente. Os pesos têm seus valores ajustados em um processo de aprendizado e codificam o conhecimento adquirido pela rede (Braga et al., 2007).'' \cite{faceli2023aprendizado}.
\end{quote}

\section{Componentes básicos de uma RNA}

Os componentes básicos de uma RNA são \textbf{arquitetura} e \textbf{aprendizado}. ``Enquanto a arquitetura está relacionada com o
tipo, o número de unidades de processamento e a forma como os neurônios estão conectados, o aprendizado diz respeito às regras utilizadas para ajustar os pesos da rede e à informação que é utilizada por essas regras.'' \cite{faceli2023aprendizado}

\textbf{O neurônio artificial é a unidade de processamento e componente fundamental da arquitetura de uma RNA.} Ele possui \textbf{terminais de entrada} que recebem os valores, uma \textbf{função de ativação}, que é uma função matemática que realiza o processamento desses valores já ponderados, e um \textbf{terminal de saída} que corresponde à resposta do neurônio, alusivos, respectivamente, aos dendritos, corpo celular e axônios de um neurônio biológico. A cada terminal de entrada corresponde
um \textbf{peso sináptico}, sendo a \textbf{entrada total}, sobre a qual a função de ativação é aplicada, definida pelo somatório de cada um dos valores de entrada multiplicado pelo peso vinculado à conexão respectiva. Os terminais de entrada podem ter pesos positivos, negativos ou zero, neste caso indicativo de que nenhuma conexão foi associada.

% TODO incluir uma imagem ilustrativa de um neurônio artificial:

Em relação à \textbf{função de ativação}, dentre as propostas mais comuns, destacam-se as seguintes: linear, limiar, sigmoidal, tangente hiperbólica, gaussiana e linear retificada (ReLU). Em qualquer caso, ela receberá a entrada total e retornará um valor que será a saída do neurônio, definindo, por conseguinte, se ele será ou não ativado.

\begin{quote}
``O uso da \textbf{função linear} identidade {[}\ldots{]} implica retornar como saída o valor de \(u\) {[}ou seja, a própria entrada total{]}. Na \textbf{função limiar} {[}\ldots{]}, o valor do limiar define quando o resultado da função limiar será igual a 1 ou 0 (alternativamente, {[}\ldots{]} -1). Quando a soma das entradas recebidas ultrapassa o limiar estabelecido, o neurônio torna-se ativo (saída +1). Quanto maior o valor do limiar, maior tem que ser o valor da entrada total para que o valor de saída do neurônio seja igual a 1. A \textbf{função sigmoidal} {[}\ldots{]} representa uma aproximação contínua e diferenciável da função limiar. A sua saída é um valor no intervalo aberto (0, 1), podendo apresentar diferentes inclinações. 

A \textbf{função tangente hiperbólica} {[}\ldots{]} é uma variação da função sigmoidal que utiliza o intervalo aberto (-1, +1) para o valor de saída. Outra função utilizada com frequência, também contínua e diferenciável, é a \textbf{função gaussiana} {[}\ldots{]}. Mais recentemente, com a popularização das redes profundas, passou a ser cada vez mais utilizada a \textbf{função linear retificada}, também conhecida como ReLU (do inglês \emph{Rectified Linear Unit}) {[}\ldots{]}. Essa função retorna 0 se recebe um valor negativo ou o próprio valor, no caso contrário. Junto com suas variações, ela tem apresentado bons resultados
em várias aplicações.'' \cite{faceli2023aprendizado}
\end{quote}

%TODO A seguir, imagens ilustrativas do cálculo da entrada total de um
%neurônio artificial e das precitadas funções de ativação:

%Figura 24 --- Entrada total de um neurônio artificial (FACELI et al.,
%2023, p.~104).

%Figura 25 --- Funções de ativação (FACELI et al., 2023, p.~104).

\section{Arquitetura}

As redes neurais artificiais tem seus neurônios organizados em \textbf{camadas} que definem o \textbf{padrão arquitetural} da rede. Na forma mais simples, composta por \textbf{uma única camada}, os neurônios recebem os dados diretamente em seus terminais de entrada, correspondendo ela própria à camada de saída. Nas redes \textbf{multicamadas}, que possuem camadas \textbf{intermediárias, escondidas ou ocultas}, o fluxo da informação entre as camadas pode ser unidirecional (redes \textbf{\emph{feed-forward}}) ou com \textbf{retroalimentação (\emph{feedback})} (redes \textbf{recorrentes ou com retropropagação}), isto é, um ou mais terminais de entrada de um ou mais neurônios recebem a saída de neurônios da mesma camada, de
camada posterior ou mesmo a sua própria saída. ``O número de camadas, o número de neurônios em cada camada, o grau de conectividade e a presença ou não de conexões com retropropagação definem a \textbf{topologia} de uma RNA.'' \cite{faceli2023aprendizado}.

%TODO Ilustrando arquiteturas multicamadas sem e com retroalimentação, vejamos
%a imagem a seguir:

%Figura 26 --- RNA multicamadas sem e com retroalimentação (FACELI et
%al., 2023, p.~106).

\section{Aprendizado}

Essencialmente, a capacidade de aprendizado está relacionada com o \textbf{ajuste dos parâmetros} da RNA, isto é, ``{[}\ldots{]} a
\textbf{definição dos valores dos pesos associados às conexões da rede} que fazem com que o modelo obtenha melhor desempenho, geralmente medido pela acurácia preditiva.'' \cite{faceli2023aprendizado}.

Isso é feito através de ``{[}\ldots{]} um \textbf{conjunto de regras bem definidas} que especificam quando e como deve ser alterado o valor de cada peso'' (FACELI et al., 2023, p.~106), implementadas por \textbf{algoritmos de treinamento}.

Os principais algoritmos são os seguintes: \textbf{(1) correção de erro}: o ajuste é feito para minimizar os erros, geralmente, mediante o aprendizado supervisionado; \textbf{(2) Hebbiano}: inspirado na aprendizagem Hebbiana --- não supervisionada ---, que preconiza o fortalecimento de conexões que, com frequência, são ativadas simultaneamente; \textbf{(3) competitivo}: os neurônios competem entre si (aprendizado não supervisionado); e \textbf{(4) termodinâmico}: inspirado na aprendizagem de Boltzmann, é um algoritmo estocástico que se baseia em princípios da Física (termodinâmica) e busca o ``equilíbrio térmico'' da rede.
