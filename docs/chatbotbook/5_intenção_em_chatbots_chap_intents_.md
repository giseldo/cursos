# Intenção em Chatbots {#chap:intents}

## Introdução

Os Intents representam a intenção ou o propósito por trás da mensagem de
um usuário ao interagir com o chatbot. Em termos mais simples, é o que o
usuário deseja que o chatbot faça ou sobre o que ele quer saber.

## Definição e Propósito {#sec:intents_definicao}

Um intent é usado para identificar programaticamente a intenção da
pessoa que está usando o chatbot. O chatbot deve ser capaz de executar
alguma ação com base no \"intent\" que detecta na mensagem do usuário.
Cada tarefa que o chatbot deve realizar define um intent.

## Exemplos de Intents {#sec:intents_exemplos}

A aplicação prática dos intents varia conforme o domínio do chatbot:

- Para um chatbot de uma loja de moda, exemplos de intents seriam
  `busca de um produto` (quando um usuário quer ver produtos) e
  `endereço loja` (quando um usuário pergunta sobre lojas).

- Em um chatbot para pedir comida, `consultar preços` e
  `realizar pedido` podem ser intents distintos.

## Detecção de Intent {#sec:intents_deteccao}

Detectar o intent da mensagem do usuário é um problema conhecido de
aprendizado de máquina, realizado por meio de uma técnica chamada
classificação de texto. O objetivo é classificar frases em múltiplas
classes (os intents). O modelo de aprendizado de máquina é treinado com
um conjunto de dados que contém exemplos de mensagens e seus intents
correspondentes. Após o treinamento, o modelo pode prever o intent de
novas mensagens que não foram vistas antes.

## Utterances (Expressões do Usuário) {#sec:intents_utterances}

Cada intent pode ser expresso de várias maneiras pelo usuário. Essas
diferentes formas são chamadas de *utterances* ou expressões do usuário.

Por exemplo, para o intent `realizar pedido`, as utterances poderiam ser
\"\"Eu gostaria de fazer um pedido\", \"Quero pedir comida\", \"Como
faço para pedir?\", etc. Cada uma dessas expressões representa a mesma
intenção, mas com palavras diferentes. O modelo de aprendizado de
máquina deve ser capaz de reconhecer todas essas variações como
pertencentes ao mesmo intent.

É sugerido fornecer um número ótimo de utterances variadas por intent
para garantir um bom treinamento do modelo de reconhecimento.

## Entities (Entidades) {#sec:intents_entities}

Os Intents frequentemente contêm metadados importantes chamados
*Entities*. Estas são palavras-chave ou frases dentro da utterance do
usuário que ajudam o chatbot a identificar detalhes específicos sobre o
pedido, permitindo fornecer informações mais direcionadas. Por exemplo,
na frase \"Eu quero pedir uma pizza de calabreza com borda rechada\", as
entidades podem incluir:

- O **Intent** é `realizar pedido`.

- As **Entities** podem ser: `pizza`, `calabreza`, `borda recheada`.

As entidades extraídas permitem ao chatbot refinar sua resposta ou ação.

## Treinamento do Bot {#sec:intents_treinamento}

O processo de treinamento envolve a construção de um modelo de
aprendizado de máquina. Este modelo aprende a partir do conjunto
definido de intents, suas utterances associadas e as entidades anotadas.
O objetivo do treinamento é capacitar o modelo a categorizar
corretamente novas utterances (que não foram vistas durante o
treinamento) no intent apropriado e a extrair as entidades relevantes.

## Pontuação de Confiança (Confidence Score) {#sec:intents_confianca}

Quando o chatbot processa uma nova mensagem do usuário, o modelo de
reconhecimento de intent não apenas classifica a mensagem em um dos
intents definidos, mas também fornece uma *pontuação de confiança*
(geralmente entre 0 e 1). Essa pontuação indica o quão seguro o modelo
está de que a classificação está correta. É comum definir um *limite
(threshold)* de confiança. Se a pontuação do intent detectado estiver
abaixo desse limite, o chatbot pode pedir esclarecimentos ao usuário em
vez de executar uma ação baseada em uma suposição incerta.

## Uso Prático e Análise {#sec:intents_uso_pratico}

Uma vez que um intent é detectado com confiança suficiente, o chatbot
pode executar a ação correspondente. Isso pode envolver consultar um
banco de dados, chamar uma API externa, fornecer uma resposta estática
ou iniciar um fluxo de diálogo mais complexo. Além disso, a análise dos
intents mais frequentemente capturados fornece insights valiosos sobre
como os usuários estão interagindo com o chatbot e quais são suas
principais necessidades. Essas análises são importantes tanto para a
otimização do bot quanto para as decisões de negócio.

## Resumo e Relação com Outros Conceitos {#sec:intents_resumo}

Em resumo, Intents são um conceito central na arquitetura de chatbots
modernos baseados em NLU (Natural Language Understanding). Eles
representam o objetivo do usuário e permitem que o chatbot compreenda a
intenção por trás das mensagens para agir de forma adequada. Os Intents
estão intrinsecamente ligados a outros conceitos fundamentais:

- **Entities:** Fornecem os detalhes específicos dentro de um intent.

- **Utterances:** São as diversas maneiras como um usuário pode
  expressar um mesmo intent.

- **Actions/Responses:** São as tarefas ou respostas que o chatbot
  executa após identificar um intent.

A definição cuidadosa, o treinamento robusto e o gerenciamento contínuo
dos intents são cruciais para a eficácia, a inteligência e a qualidade
da experiência do usuário oferecida por um chatbot.
