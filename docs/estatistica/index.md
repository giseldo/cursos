# Estatística e AM com Python

:::info
<div style="display: flex; gap: 10px; align-items: center;">
  <img src="./fig/esquerda.jpg" style="width:5%" alt="image" />
  <img src="./fig/direita.jpg" style="width:5%" alt="image" />
  Você pode usar as setas do teclado para navegar entre as seções.
</div>
:::


Em aprendizagem de máquina utiliza-se um _conjunto de dados_ para realizar previsões.

## Conjunto de dados

Um conjunto de dados geralmente é representado em forma de uma matriz ou tabela, onde a primeira linha da tabela representa o nome do atributo. E as linhas subsequentes os valores para cada um dos atributos.

Veja na Tabela 1 um conjunto de dados com pessoas em uma sala de aula. Os atributos deste conjunto de dados são: nome, idade e perfil. Neste conjunto de dados existem cinco observações, ou cinco linhas, cada observação é uma pessoa na sala de aula.

Tabela 1 - Conjunto de dados fictício.

|Nome|Idade|Perfil|
|---|---|---|
|Giseldo|40|professor|
|Alex|14|aluno|
|Alana|14|aluno|
|Gisella|15|aluno|
|Alice|16|aluno|

:::info Sinônimos

- Conjunto de dados:
    - Tabela
    - Matriz

- Atributo:
    - Coluna
    - Característica

- Observações:
    - Linhas

:::

## Tipo do atributo

Um atributo pode ser do tipo _numérico_ ou _categórico_. Se ele for numérico, pode ser ainda _contínuo_ (representando uma medida, ex: peso, altura) ou _discreto_ (representando uma contagem, por exemplo, idade). Se for categórico, pode ser _nominal_ (por exemplo, nome ou cidade) ou _ordinal_ (pode ser ordenado, nível de escolaridade). Ainda existe um tipo especial de atributo categórico que é _binário_. O tipo categórico binário pode assumir dois valores, por exemplo, 0 ou 1, True ou False e outros semelhantes.

Apresentar os tipos dos atributos do conjunto de dados em uma outra tabela, conforme a Tabela 2, é uma boa prática em artigos científicos. Além disso, conhecer o tipo dos atributos é necessário para criar resumos e visualizações dos dados. 

Tabela 2 - Uma tabela com os tipos dos dados do conjunto de dados fictício.

|Atributo|Tipo|
|---|---|
|nome|qualitativo nominal|
|idade|quantitativo discreto|
|perfil|qualitativo binário (ou aluno ou professor)|

:::info Sinônimos

- Numérico:
    - Quantitativo

- Categórico
    - Qualitativo

:::


## Medidas de localização

Caso o tipo de um um atributo seja númerico, pode-se resumir este atributo com as medidas de localização: _média_, _mediana_, ou _moda_ (e mais alguns outros).

### Média

A média é calculada somando os valores e dividido pela quantidade de observações. A seguir um código python para calcular a média de um conjunto de números

```Python
idade = [40, 14, 14, 15, 16]
mean(idade) # 19,8
```

$\sqrt{3x-1}+(1+x)^2$

### Mediana

A mediana é calculanda ordenando os valores e selecionando a observação que está no ponto médio, se o conjunto de dados tiver número impar de valores será o valor do meio, se a quantidade de observações for par, será a soma dos dois valores do meio.

<!-- TODO: Explicar melhor --> 
<!-- TODO: Ordenar a idade em python--> 

```Python
idade = [14, 14, 15, 16, 40]
median(idade) # 15
```

### Moda

A terceira medida de localização é a moda.

```Python
idade = [14, 14, 15, 16, 40]
moda(idade) # 14
```

### Qual utilizar? Média, mediana ou moda?

Para o atributo _idade_ temos três valores diferentes para as medidas de localização media, mediana e moda. Veja os Valores na Tabela 3. Qual o valor deve ser utilizado para representar este atributo em um artigo 
científico? Isso depende de alguns critérios que serão apresentados a seguir. 

> Tabela 3 - Medidas de localidade para o atributo idade
>
> |Medida de localidade do atributo idade|Valor|
> |---|---|
> Média|29,5| 
> Moda|15|
> Mediana|14|

A média é sensível a _outliers_. Outliers são valores que podem ser muitos distantes da grande maiora das observações. Pode ser um erro ou não. Por exemplo, No nosso conjunto de dados fictício (Tabela 1), nota-se que a idade do professor, neste caso é bem superior a idade dos alunos da turma.

Se um dos objetivos for representar a idade dos alunos com a média, a mesma seria de 21,25 anos, ou seja em média uma observação (ou aluno) tem 21,25 anos. Como temos o valor da observação professor no conjunto, com 40 anos, isso eleva a média. Porém analisando os dados concluí-se que não tem nenhum aluno com 21 anos no nosso conjunto. Portanto utilizar a mediana ou a moda seria mais apropriado. 

Utilizando a mediana, portanto o número seria de 15, ou seja a medida de localização do conjunto seria de 15 anos. E utilizando a moda 14 anos. Valores mais coerentes com o objetivo de representar a média da turma, ou dos estudantes neste caso.

Outras técnicas ainda podem ser utilizadas: tais como remover os outliers do nosso conjunto e calcular a média novamente. 

