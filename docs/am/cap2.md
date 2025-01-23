---
title: Preparação dos dados
---

# Preparação dos dados

Os dados constituem o substrato fundamental para o funcionamento dos algoritmos de aprendizado de máquina. Contudo, os dados em sua forma bruta frequentemente apresentam problemas como inconsistência, redundância, desbalanceamento, ruídos e incompletudes. Tais problemas podem surgir devido a falhas na integração de múltiplas fontes de dados, à grande quantidade de objetos ou atributos, à falta de uniformidade na distribuição dos dados, à ausência de valores, à incompatibilidade de formatos ou a erros ocorridos em qualquer fase do ciclo de vida dos dados, como geração, coleta, armazenamento, ou durante a alimentação dos dados no algoritmo. Nesses cenários, técnicas de pré-processamento tornam-se essenciais para corrigir esses problemas e preparar os conjuntos de dados de forma adequada para serem processados pelos algoritmos de aprendizado de máquina \cite{hastie2009elements}. Além disso, a caracterização e exploração inicial dos dados por meio de estatísticas descritivas e técnicas de visualização são práticas recomendadas para melhor compreensão das características dos dados.

Existem diversas técnicas de pré-processamento de dados, mas não existe uma hierarquia fixa ou preferência na ordem de execução entre elas. A escolha de uma técnica sobre outra depende do contexto específico do problema a ser resolvido, e é comum que o pré-processamento envolva múltiplas etapas ou a repetição de técnicas ao longo do processo \cite{faceli2023aprendizado}. Esse conjunto de práticas, quando aplicado desde o início até a fase final de modelagem, é frequentemente referido como aprendizado de máquina de ponta a ponta, englobando não apenas o pré-processamento, mas também a modelagem e o pós-processamento dos dados.

Entre as principais tarefas de pré-processamento estão a integração de dados, eliminação manual de atributos, amostragem de dados, balanceamento de dados, limpeza de dados, redução de dimensionalidade e transformação de dados \cite{han2011data}. Essas tarefas são parte integrante do processo de Mineração de Dados, cujo objetivo é extrair conhecimento novo, útil e relevante de um conjunto de dados \cite{fayyad1996data}. Enquanto a Mineração de Dados foca no processo de extração de conhecimento, a Ciência de Dados tem um enfoque mais amplo, concentrando-se na manipulação de dados de diferentes tipos e na avaliação do impacto e relevância dos dados para aplicações práticas \cite{provost2013data}.

Dessa forma, o pré-processamento de dados é uma etapa que influencia diretamente a eficácia dos algoritmos de aprendizado de máquina. Um pré-processamento adequado não só melhora a qualidade dos dados, mas também potencializa a capacidade dos modelos em aprender padrões significativos, evitando problemas comuns como overfitting e underfitting, e garantindo que os resultados obtidos sejam robustos e generalizáveis \cite{goodfellow2016deep}.

## Atributos

No contexto de aprendizado de máquina, a natureza dos atributos dos dados é um fator determinante na escolha dos métodos e algoritmos apropriados para análise. Atributos podem ser classificados como \textbf{quantitativos}, quando representam quantidades, ou \textbf{qualitativos}, quando descrevem qualidades. Atributos quantitativos, também conhecidos como numéricos, são essenciais em muitos modelos preditivos, pois permitem operações aritméticas e análises estatísticas complexas \cite{murphy2012machine}. Por outro lado, atributos qualitativos, também chamados de simbólicos ou categóricos, associam o objeto a uma categoria ou classe específica bastante utilizado em tarefas de classificação \cite{domingos2015master}.

Os atributos quantitativos são geralmente divididos em duas categorias principais: \textbf{contínuos} e \textbf{discretos}. Atributos contínuos são aqueles que podem assumir um número infinito de valores possíveis, geralmente representados por números reais, e frequentemente resultam de medições, como peso ou temperatura, com uma unidade de medida associada \cite{hastie2009elements}. Atributos discretos, em contraste, têm um conjunto finito ou infinito, mas contável, de valores possíveis, como contagens inteiras. A habilidade de ordenar e realizar operações aritméticas em atributos quantitativos os torna particularmente valiosos em modelos de regressão e outras técnicas estatísticas \cite{bishop2006pattern}.

Atributos qualitativos, por sua vez, consistem tipicamente em uma quantidade finita de símbolos ou nomes. Embora possam ser codificados numericamente para facilitar o processamento computacional, não faz sentido realizar operações aritméticas nesses valores, uma vez que eles não possuem um significado numérico inerente \cite{aggarwal2015data}. No entanto, em muitos casos, é possível estabelecer uma ordem entre esses atributos, como em classificações hierárquicas, o que permite a aplicação de métodos estatísticos não paramétricos para análise \cite{provost2013data}.

A escala de medição de um atributo define as operações que podem ser realizadas sobre seus valores e é classificada em nominal, ordinal, intervalar e racional. Atributos nominais, que são qualitativos, não podem ser ordenados e apenas permitem operações de igualdade e desigualdade \cite{faceli2023aprendizado}. Atributos ordinais permitem ordenação e comparação usando operadores lógicos, enquanto atributos intervalares, que são quantitativos, medem variações em intervalos e expressam ordem e magnitude, mas têm um ponto zero arbitrário. Atributos racionais, por outro lado, têm um zero absoluto e são associados a uma unidade de medida, permitindo a realização de operações aritméticas completas, o que lhes confere o maior valor semântico \cite{garcia2016data}.

\begin{figure}
    \centering
    \includegraphics[width=0.7\linewidth]{figs/atributos.png}
    \caption{Categoria dos Atributos}
    \label{fig:enter-label}
\end{figure}

## Exploração de Dados

A exploração de dados é uma etapa crucial no processo de análise, cujo objetivo principal é a extração de informações úteis do conjunto de dados. Esta fase orienta a escolha da abordagem mais adequada para resolver o problema em questão, incluindo a seleção das técnicas de pré-processamento e algoritmos de aprendizado de máquina \cite{fayyad1996data}. A estatística descritiva é uma ferramenta essencial nesse contexto, pois permite resumir quantitativamente as principais características de um conjunto de dados. As medidas estatísticas obtidas podem ser vistas como estimativas dos parâmetros estatísticos da distribuição que gerou os dados, fornecendo uma base sólida para a modelagem subsequente \cite{hastie2009elements}.

Entre as diversas métricas disponíveis, é possível obter informações sobre a frequência de um determinado valor, seja ele numérico ou simbólico, no conjunto de dados. Além disso, as medidas de localização ou tendência central, como moda, média, mediana, quartis e percentis, são fundamentais para entender o comportamento central dos dados \cite{bishop2006pattern}. Medidas de dispersão, como intervalo, variância e desvio padrão, fornecem insights sobre o grau de espalhamento dos dados em torno de um valor central, enquanto as medidas de distribuição ou formato, como obliquidade e curtose, revelam a simetria e o achatamento da distribuição dos dados em relação à distribuição normal \cite{murphy2012machine}.

Apesar de a maioria dos conjuntos de dados utilizados em aprendizado de máquina apresentar mais de um atributo, análises univariadas, ou seja, realizadas em cada atributo individualmente, podem oferecer informações valiosas sobre a natureza dos dados \cite{domingos2015master}. Assim, as estatísticas descritivas não devem ser calculadas apenas para dados multivariados, mas também para dados univariados, pois isso permite uma compreensão mais profunda e detalhada dos padrões subjacentes nos dados.

### Medidas

#### Frequência 

A frequência refere-se à quantidade proporcional de vezes que um determinado valor, seja ele numérico ou simbólico, é encontrado em um conjunto de dados. Veja na Figura~\ref{fig:freq} um exemplo do cálculo da frequência em Python.

```Python
import pandas as pd

# Exemplo de DataFrame
data = {
    'Categoria': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'B', 'C', 'C']
}

df = pd.DataFrame(data)

# Calculando a frequência dos valores na coluna 'Categoria'

frequencia = df['Categoria'].value_counts()

# Exibindo o resultado

print(frequencia)
```

#### Localização ou Tendência Central

As medidas de localização ou tendência central são utilizadas para identificar pontos de referência nos dados. A moda, por exemplo, é o valor que aparece com maior frequência no conjunto de dados e é frequentemente utilizada para atributos qualitativos. A média, que é a soma dos valores dividida pelo número total de elementos, é uma das medidas mais comuns para atributos quantitativos, embora seja sensível a outliers. A mediana, que divide o conjunto de dados em duas partes iguais, é menos influenciada por valores extremos e oferece uma visão mais robusta da centralidade dos dados \cite{provost2013data}. Quartis e percentis dividem o conjunto de dados em partes iguais, proporcionando uma compreensão detalhada da distribuição dos valores. Veja na Figura~\ref{fig:media} um exemplo em Python.

```Python
import pandas as pd
from scipy import stats

# Exemplo de DataFrame
data = {
    'Valores': [10, 20, 20, 30, 40, 50, 60, 70, 80, 20]
}

df = pd.DataFrame(data)

# Calculando a média
media = df['Valores'].mean()

# Calculando a mediana
mediana = df['Valores'].median()

# Calculando a moda
moda = df['Valores'].mode()[0]  # mode() retorna uma Series, pegamos o primeiro valor

# Exibindo os resultados
print(f"Média: {media}")
print(f"Mediana: {mediana}")
print(f"Moda: {moda}")
```

#### Dispersão ou Espalhamento

As medidas de dispersão ou espalhamento determinam se os valores estão amplamente espalhados ou concentrados em torno de um valor central, como a média \cite{garcia2016data}. O intervalo, que é a diferença entre o maior e o menor valor, revela a amplitude do conjunto de dados. A variância mede o desvio médio dos valores em relação à média, sendo altamente sensível a outliers. Para mitigar esse problema, o desvio padrão, que é a raiz quadrada da variância, é frequentemente utilizado como uma medida mais robusta de dispersão. Veja na Figura~\ref{fig:intervalo}

```Python
import pandas as pd

# Exemplo de DataFrame
data = {
    'Valores': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

df = pd.DataFrame(data)

# Calculando o intervalo (amplitude)
intervalo = df['Valores'].max() - df['Valores'].min()

# Calculando a variância
variancia = df['Valores'].var()

# Calculando o desvio padrão
desvio_padrao = df['Valores'].std()

# Exibindo os resultados
print(f"Intervalo (Amplitude): {intervalo}")
print(f"Variância: {variancia}")
print(f"Desvio Padrão: {desvio_padrao}")
```

### Distribuição ou Formato

As medidas de distribuição ou formato revelam como os dados estão distribuídos. A obliquidade indica a simetria da distribuição, enquanto a curtose descreve o achatamento da distribuição em relação à normal \cite{hastie2009elements}. Uma distribuição normal tem obliquidade e curtose iguais a zero; distribuições assimétricas ou com diferentes graus de achatamento apresentarão valores positivos, ou negativos para essas métricas. Veja na Figura~\ref{fig:obliquidade}.

```Python
import numpy as np
from scipy.stats import skew, kurtosis

# Exemplo de distribuição normal
data = np.random.normal(0, 1, 1000)

# Cálculo da obliquidade e curtose
obliquidade = skew(data)
curtose = kurtosis(data)

print(f"Obliquidade: {obliquidade:.4f}")
print(f"Curtose: {curtose:.4f}")
```
