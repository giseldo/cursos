# Expressões Regulares

## Introdução

Expressões regulares, frequentemente abreviadas como regex, são
sequências de caracteres que definem padrões de busca. Elas são
utilizadas em chatbots para diversas tarefas relacionadas ao
processamento e à análise de texto fornecido pelos usuários. Algumas das
aplicações incluem:

- **Extração de entidades:** Identificação e extração de informações
  específicas, como endereços de e-mail, números de telefone, datas e
  outros dados estruturados presentes na entrada do usuário.

- **Validação de entradas do usuário:** Verificação se a entrada do
  usuário corresponde a um formato esperado, como datas em um formato
  específico (DD/MM/AAAA), códigos postais ou outros padrões
  predefinidos.

- **Detecção de Intenção:** Detecção de comandos específicos inseridos
  pelo usuário, como `/ajuda`, `/iniciar` ou palavras-chave que indicam
  uma intenção específica.

- **Limpeza de texto:** Remoção de ruídos e elementos indesejados do
  texto, como tags HTML, espaços em branco excessivos ou caracteres
  especiais que podem interferir no processamento subsequente.

- **Tokenização simples:** Embora métodos mais avançados sejam comuns em
  PLN, regex pode ser usada para dividir o texto em unidades menores
  (tokens) com base em padrões simples.

Essas tarefas são fundamentais para garantir que o chatbot possa
interpretar e responder adequadamente às entradas dos usuários,
especialmente em cenários onde a informação precisa ser estruturada ou
verificada antes de ser processada por modelos de linguagem mais
complexos.

## Fundamentos do Módulo `re` em Python

O módulo `re` em Python é a biblioteca padrão para trabalhar com
expressões regulares. Ele fornece diversas funções que permitem realizar
operações de busca, correspondência e substituição em strings com base
em padrões definidos por regex. Algumas das funções mais utilizadas
incluem:

- `re.match(pattern, string)`: Tenta encontrar uma correspondência do
  padrão no *início* da string. Se uma correspondência for encontrada,
  retorna um objeto de correspondência; caso contrário, retorna `None`.

- `re.search(pattern, string)`: Procura a primeira ocorrência do padrão
  em *qualquer posição* da string. Retorna um objeto de correspondência
  se encontrado, ou `None` caso contrário.

- `re.findall(pattern, string)`: Encontra *todas* as ocorrências não
  sobrepostas do padrão na string e as retorna como uma lista de
  strings.

- `re.sub(pattern, repl, string)`: Substitui todas as ocorrências do
  padrão na string pela string de substituição `repl`. Retorna a nova
  string resultante.

### Exemplo Básico: Extração de E-mails

Um caso de uso comum em chatbots é a extração de endereços de e-mail do
texto fornecido pelo usuário. O seguinte exemplo em Python demonstra
como usar `re.findall` para realizar essa tarefa:

``` {#lst:extracao_email .python language="Python" caption="Extração de e-mails com regex" label="lst:extracao_email"}
import re

texto = "Entre em contato em exemplo@email.com ou suporte@outroemail.com."
padrao = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
emails = re.findall(padrao, texto)
print(emails)
```

A saída deste código será:

    ['exemplo@email.com', 'suporte@outroemail.com']

Este exemplo ilustra a eficácia das regex para identificar e extrair
informações específicas de um texto.

## Sintaxe de Expressões Regulares

A sintaxe das expressões regulares consiste em uma combinação de
caracteres literais (que correspondem a si mesmos) e metacaracteres, que
possuem significados especiais e permitem definir padrões de busca mais
complexos. Alguns dos metacaracteres mais importantes incluem:

## Casos de Uso Específicos em Chatbots

As expressões regulares podem ser aplicadas em uma variedade de cenários
no desenvolvimento de chatbots. A seguir, apresentamos alguns casos de
uso comuns com exemplos práticos em Python.

### Validação de Datas

Chatbots que lidam com agendamentos ou reservas frequentemente precisam
validar se a data fornecida pelo usuário está em um formato correto. O
seguinte exemplo demonstra como validar datas no formato DD/MM/AAAA:

``` {#lst:validacao_data .python language="Python" caption="Validação de datas com regex" label="lst:validacao_data"}
import re

padrao_data = r'\b\d{2}/\d{2}/\d{4}\b'
datas_teste = ["31/12/2020", "1/1/2021", "2023-05-10", "25/06/2025 10:00"]

for data in datas_teste:
    if re.match(padrao_data, data):
        print(f"'{data}' é uma data válida no formato DD/MM/AAAA.")
    else:
        print(f"'{data}' não é uma data válida no formato DD/MM/AAAA.")
```

A saída deste código ilustra quais das strings de teste correspondem ao
padrão de data especificado.

### Análise de Comandos

Em interfaces de chatbot baseadas em texto, os usuários podem interagir
através de comandos específicos, como `/ajuda` ou `/iniciar`. As regex
podem ser usadas para detectar esses comandos de forma eficiente:

``` {#lst:analise_comando .python language="Python" caption="Análise de comandos com regex" label="lst:analise_comando"}
import re

padrao_comando = r'^/\w+'
comandos_teste = ["/ajuda", "/iniciar", "ajuda", "iniciar/"]

for comando in comandos_teste:
    if re.match(padrao_comando, comando):
        print(f"'{comando}' é um comando válido.")
    else:
        print(f"'{comando}' não é um comando válido.")
```

Este exemplo mostra como identificar strings que começam com uma barra
seguida por um ou mais caracteres alfanuméricos.

### Tokenização Simples

Embora para tarefas complexas de PLN sejam utilizadas técnicas de
tokenização mais avançadas, as regex podem ser úteis para realizar uma
tokenização básica, dividindo o texto em palavras ou unidades menores
com base em padrões de separação:

``` {#lst:tokenizacao_simples .python language="Python" caption="Tokenização simples com regex" label="lst:tokenizacao_simples"}
import re

texto = "Olá, como vai você?"
tokens = re.split(r'\W+', texto)
print(tokens)
```

A saída será uma lista de strings, onde o padrão `\W+` corresponde a um
ou mais caracteres não alfanuméricos, utilizados como delimitadores.

### Limpeza de Texto

Chatbots podem precisar processar texto que contém elementos
indesejados, como tags HTML. As regex podem ser usadas para remover
esses elementos:

``` {#lst:limpeza_html .python language="Python" caption="Limpeza de texto removendo tags HTML" label="lst:limpeza_html"}
import re

texto_html = "<p>Este é um parágrafo com <b>texto em negrito</b>.</p>"
texto_limpo = re.sub(r'<[^>]+>', '', texto_html)
print(texto_limpo)
```

## Aplicação em Frameworks de Chatbot

Frameworks populares para desenvolvimento de chatbots, como Rasa,
frequentemente integram o uso de expressões regulares para aprimorar a
extração de entidades. Por exemplo, em Rasa, as regex podem ser
definidas nos dados de treinamento para ajudar o sistema a reconhecer
padrões específicos como nomes de ruas ou códigos de produtos. Essa
abordagem permite melhorar a precisão do reconhecimento de entidades, um
componente crucial para a compreensão da intenção do usuário.

## Tópicos Avançados

Embora os fundamentos das regex sejam suficientes para muitas tarefas,
existem construções mais avançadas que podem ser úteis em cenários
complexos. Alguns exemplos incluem:

- **Lookaheads e Lookbehinds:** Permitem verificar se um padrão é
  seguido ou precedido por outro padrão, sem incluir esse outro padrão
  na correspondência.

- **Correspondência não-gulosa:** Ao usar quantificadores como `*` ou
  `+`, a correspondência padrão é \"gulosa\", ou seja, tenta
  corresponder à maior string possível. Adicionar um `?` após o
  quantificador (`*?`, `+?`) torna a correspondência \"não-gulosa\",
  correspondendo à menor string possível.

A exploração detalhada desses tópicos está além do escopo deste capítulo
introdutório, mas são ferramentas poderosas para lidar com padrões mais
complexos.

## Limitações e Contexto

É importante reconhecer que, apesar de sua utilidade, as expressões
regulares têm limitações significativas quando se trata de compreender a
complexidade da linguagem natural. As regex são baseadas em padrões
estáticos e não possuem a capacidade de entender o contexto, a semântica
ou as nuances da linguagem humana.

Para tarefas que exigem uma compreensão mais profunda do significado e
da intenção por trás das palavras, técnicas avançadas de Processamento
de Linguagem Natural (PLN), como modelagem de linguagem, análise de
sentimentos e reconhecimento de entidades nomeadas (NER) baseados em
aprendizado de máquina, são indispensáveis.

No contexto de um fluxo de trabalho de chatbot, as expressões regulares
são frequentemente mais eficazes nas etapas de pré-processamento, como
limpeza e validação de entradas, enquanto técnicas de PLN mais
sofisticadas são empregadas para a compreensão da linguagem em um nível
mais alto. Os capítulos posteriores deste livro abordarão essas técnicas
avançadas, incluindo o uso de Modelos de Linguagem Grandes (LLMs) e
Retrieval-Augmented Generation (RAG), que complementam o uso de regex,
permitindo a construção de chatbots mais inteligentes e contextualmente
conscientes.

## Conclusão

As expressões regulares representam uma ferramenta essencial para o
processamento de texto em chatbots, oferecendo uma maneira eficaz de
extrair informações específicas, validar formatos de entrada e realizar
tarefas básicas de limpeza de texto. Através do módulo `re` em Python,
os desenvolvedores têm à disposição um conjunto de funcionalidades
poderosas para manipular strings com base em padrões definidos.

No entanto, é crucial entender as limitações das regex, especialmente no
que diz respeito à compreensão da linguagem natural em sua totalidade.
Para tarefas que exigem análise semântica e contextual, técnicas
avançadas de PLN são necessárias. As expressões regulares, portanto,
encontram seu melhor uso como parte de um fluxo de trabalho mais amplo,
onde complementam outras abordagens para criar chatbots robustos e
eficientes.

Encorajamos o leitor a praticar a criação de diferentes padrões de regex
e a experimentar com os exemplos fornecidos neste capítulo. A
familiaridade com as expressões regulares é uma habilidade valiosa para
qualquer pessoa envolvida no desenvolvimento de chatbots e no
processamento de linguagem natural em geral.
