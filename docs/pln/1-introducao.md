
# Introdução

Processamento de Linguagem Natural (PLN) é um campo da linguística e da aprendizagem de máquina focada em entender a linguagem humana. O objetivo das tarefas de PLN não é apenas entender palavras soltas individualmente, mas ser capaz de entender o contexto dessas palavras.

A seguir uma lista de tarefas comuns de NLP, com alguns exemplos.

![alt text](fig/tasks_pln.png)

* Classificação de sentenças 
  * Capturar o sentimento de uma sentença, relacionado por exemplo a revisão de determinado produto; (_sentiment_analisys_)
  * Detectar se um email é spam ou não; (_text_classification_)
  * Determinar se uma sentença é gramaticalmente correta; 
  * Determinar se duas sentencas são logicamente relacionadas ou não.

* Classificação de cada palavra em uma sentença 
  * Identificar os componentes gramaticais de uma sentença, por exemplo, substantivo, verbo, adjetivo;
  * Identificar as entidades nomeadas, por exemplo, pessoa, local, organização; (*named_entity_recognition*)

* Extrair uma resposta de um texto 
  * Dada uma pergunta e um contexto, extrair a resposta baseada na informação passada no contexto. (*question_answering*)

* Gerar uma nova sentença a partir de uma entrada de texto
  * Traduzir a sentença para outro idioma (_translation_)
  * resumir um texto (_sumarization_)

* Geração de conteúdo textual 
  * Completar um trecho com autogeração textual (_fill-mask_)
  * Preencher as lacunas em um texto com palavas mascaradas (_fill-mask_)

PLN não se limita ao texto escrito. Também engloba desafios complexos nos campos de reconhecimento de discurso e visão computacional, tal como a geração de transcrição de uma amostra de áudio ou a descrição de uma imagem.

## Por que isso é desafiador?

Os computadores não processam a informação da mesma forma que os seres humanos. Por exemplo, quando nós lemos a sentença "estou com fome", nós podemos facilmente entender seu significado. Similarmente, dada duas sentenças como "Estou com disposição" e "Estou alegre", nós somos capazes de facilmente determinar quão similares elas são. Para modelos de Aprendizagem de Máquina (ML), tarefas como essas são mais difíceis. O texto precisa ser processado de um modo que possibilite ao modelo aprender. E porque a linguagem é complexa, nós precisamos pensar cuidadosamente como esse processamento tem que ser feito. Várias formas de representação de texto existem. 

## Empresas

Aqui estão algumas empresas e organizações usando a Hugging Face e os modelos `transformers`. Estas empresas também contribuem de volta para a comunidade compartilhando seus modelos.

![alt text](fig/empresas.png)