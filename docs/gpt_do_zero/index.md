# LLM do Zero

Construir um LLM como o GPT-2 do zero é um projeto fascinante que vai te ensinar muito sobre redes neurais, processamento de linguagem natural e engenharia de software. 

1.  **Entendendo a Base:** O que são LLMs e o que torna o GPT-2 especial (arquitetura Transformer).
2.  **Preparando o Terreno:** Requisitos de hardware (GPU!) e software (Python, bibliotecas essenciais como PyTorch ou TensorFlow).
3.  **O Ingrediente Principal: Dados!** Como obter e preparar um grande conjunto de textos (dataset) para treinar nosso modelo (tokenização).
4.  **A Receita do GPT-2:** Detalhes da arquitetura do modelo (blocos Transformer, atenção, embeddings).
5.  **Mão na Massa: O Treinamento:** Como ensinar o modelo a aprender com os dados (função de perda, otimizador, loop de treinamento).
6.  **Dando Vida ao Modelo:** Como usar o modelo treinado para gerar textos novos.


## Entendendo a Base

O que são LLMs e o que torna o GPT-2 especial?

Imagine um LLM (Large Language Model) como um "super leitor" e "super escritor" de textos. Ele é um programa de computador treinado com uma quantidade gigantesca de texto (livros, artigos, sites da internet, etc.) para aprender os padrões da linguagem humana. O objetivo dele é entender o texto que você dá e, a partir disso, gerar textos novos que façam sentido e soem naturais, como se um humano tivesse escrito.

O GPT-2 é um exemplo famoso de LLM, criado pela OpenAI. O que o tornou especial na época (e ainda é fundamental hoje) foi o uso eficiente de uma arquitetura chamada Transformer.

Pense na arquitetura Transformer como o "cérebro" do GPT-2. A grande inovação dela é o mecanismo de atenção (attention mechanism). Sabe quando você lê uma frase longa e presta mais atenção em certas palavras para entender o contexto geral? O mecanismo de atenção faz algo parecido! Ele permite que o modelo, ao gerar a próxima palavra, dê pesos diferentes às palavras anteriores, focando nas mais relevantes para aquele momento. Isso ajuda o modelo a manter a coerência e o contexto em textos mais longos.

Por exemplo, na frase "O gato perseguiu o rato até que ele se escondeu na toca", o mecanismo de atenção ajudaria o modelo a entender que "ele" provavelmente se refere ao "rato", e não ao "gato", ao continuar a história.

Entender essa ideia básica do Transformer e da atenção é crucial, porque é a peça central que vamos precisar implementar no nosso código Python!

### Exercício

A seguir são apresentados algumas frases curtas. Em cada uma, uma palavra estará destacada. Quero que você me diga em qual(is) palavra(s) anterior(es) o modelo deveria prestar mais atenção para entender o significado da palavra destacada ou para prever a próxima palavra de forma lógica.

Frase 1:

:::info
"A cachorra correu atrás da bola, mas _ela_ era muito rápida."
:::

Para entender a quem "ela" se refere, em qual palavra anterior o modelo deveria focar sua atenção?

:::info
cachorra
:::

O modelo precisaria dar mais "peso" ou atenção à palavra "cachorra" para entender que "ela" se refere à cachorra, e não à bola. É como se o modelo "olhasse para trás" e escolhesse as pistas mais importantes.

## Preparando o terreno: os requisitos de **hardware** e **software**.

### Hardware: GPU

* **Por que uma GPU?** Treinar um LLM envolve *muitos* cálculos matemáticos acontecendo ao mesmo tempo (multiplicações de matrizes, principalmente). As GPUs (Placas de Vídeo) são projetadas para fazer exatamente esse tipo de computação paralela de forma muito mais rápida que uma CPU (o processador principal do seu PC). Sem uma GPU razoável, treinar um modelo como o GPT-2 do zero levaria um tempo impraticável (semanas, meses ou até mais!).

* **Qual GPU?** As GPUs da **NVIDIA** são as mais comuns para deep learning por causa da plataforma **CUDA**, que é muito bem suportada pelas principais bibliotecas.

* **Memória da GPU (VRAM):** Este é um fator *crítico*. A VRAM limita o tamanho do modelo e a quantidade de dados que você pode processar de uma vez (o *batch size*).
    * Para **experimentar** com modelos menores e conceitos básicos, talvez 8GB de VRAM sejam suficientes.
    * Para treinar algo um pouco mais substancial (ainda longe do GPT-2 original), 12GB, 16GB ou idealmente **24GB ou mais** de VRAM farão uma grande diferença.
    * **Realidade:** Replicar o *tamanho exato* do GPT-2 original em um PC doméstico é geralmente inviável. Precisamos ajustar nossas expectativas e focar em construir uma versão menor, mas funcional, para aprender o processo.

* **Alternativas:** Se sua GPU for limitada, existem opções como usar GPUs na nuvem (Google Colab Pro, Kaggle, AWS, GCP, Azure) ou focar em versões ainda menores do modelo.

### Software: As Ferramentas Certas

* **Linguagem:** **Python** (versão 3.x)
* **Gerenciador de Pacotes:** `pip` (que vem com Python) ou `conda` (do Anaconda/Miniconda) para instalar e gerenciar as bibliotecas.
* **Bibliotecas Essenciais:**
    * **PyTorch** ou **TensorFlow:** São as principais estruturas (frameworks) de deep learning. Elas fornecem as ferramentas para construir as camadas da rede neural, calcular gradientes automaticamente (essencial para o treinamento) e rodar tudo na GPU. Ambas são ótimas. PyTorch é frequentemente vista como um pouco mais flexível para pesquisa e construção customizada, enquanto TensorFlow (com Keras) pode ser mais direto para algumas aplicações. Podemos escolher uma para focar.
    * **Transformers (da Hugging Face):** Mesmo querendo construir "do zero", esta biblioteca é incrivelmente útil. Ela tem ótimos tokenizadores (vamos falar disso logo!), implementações de referência de modelos (bom para comparar) e muitas utilidades. Podemos decidir não usar os *modelos* prontos dela, mas o tokenizador pode nos poupar muito trabalho.
    * **Datasets (da Hugging Face):** Ajuda a baixar e pré-processar grandes conjuntos de dados de texto.
    * **NumPy:** Fundamental para operações numéricas eficientes em Python.
* **Ambiente Virtual (Recomendado):** Usar algo como `venv` (nativo do Python) ou `conda env` para criar um ambiente isolado para o projeto. Isso evita conflitos entre versões de bibliotecas de diferentes projetos.



