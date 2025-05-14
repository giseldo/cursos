# Curso Completo de Gradio 

## Índice
1. [Introdução ao Gradio](#introdução-ao-gradio)
2. [Instalação e Configuração](#instalação-e-configuração)
3. [Interface Básica](#interface-básica)
4. [Componentes e Inputs](#componentes-e-inputs)
5. [Integração com Machine Learning](#integração-com-machine-learning)
6. [Layouts e Estilização](#layouts-e-estilização)
7. [Projeto Final](#projeto-final)

## Introdução ao Gradio

### O que é Gradio?
Gradio é uma biblioteca Python open-source que permite criar interfaces web interativas para modelos de machine learning e demonstrações de código. É especialmente útil para criar demonstrações de IA, protótipos e interfaces para modelos de deep learning.

### Por que usar Gradio?
- Interface simples e intuitiva
- Suporte a múltiplos tipos de inputs/outputs
- Integração fácil com frameworks de ML (TensorFlow, PyTorch, etc.)
- Deploy rápido e compartilhamento fácil
- Comunidade ativa e documentação extensa

## Instalação e Configuração

### Requisitos
- Python 3.7+
- pip (gerenciador de pacotes Python)

### Instalação
```bash
pip install gradio
```

### Primeiro App
```python
import gradio as gr

def saudacao(nome):
    return f"Olá, {nome}!"

demo = gr.Interface(
    fn=saudacao,
    inputs="text",
    outputs="text"
)

demo.launch()
```

## Interface Básica

### Componentes Fundamentais
```python
import gradio as gr

def processar_texto(texto):
    return texto.upper()

# Interface básica
demo = gr.Interface(
    fn=processar_texto,
    inputs=gr.Textbox(label="Digite seu texto"),
    outputs=gr.Textbox(label="Resultado"),
    title="Processador de Texto",
    description="Digite um texto para convertê-lo em maiúsculas"
)

demo.launch()
```

### Exercício 1
Crie uma interface que:
1. Aceite um número como input
2. Calcule seu quadrado
3. Retorne o resultado formatado

## Componentes e Inputs

### Tipos de Inputs
```python
import gradio as gr

def processar_inputs(texto, numero, checkbox, radio):
    return f"Texto: {texto}\nNúmero: {numero}\nCheckbox: {checkbox}\nRadio: {radio}"

demo = gr.Interface(
    fn=processar_inputs,
    inputs=[
        gr.Textbox(label="Texto"),
        gr.Number(label="Número"),
        gr.Checkbox(label="Opção"),
        gr.Radio(["Opção 1", "Opção 2", "Opção 3"], label="Escolha")
    ],
    outputs="text"
)
```

### Tipos de Outputs
```python
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

def gerar_grafico():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    return plt

demo = gr.Interface(
    fn=gerar_grafico,
    inputs=None,
    outputs="plot"
)
```

### Exercício 2
Crie uma interface que:
1. Aceite uma imagem como input
2. Aplique um filtro (ex: escala de cinza)
3. Mostre a imagem original e a processada lado a lado

## Integração com Machine Learning

### Exemplo com Modelo de Classificação
```python
import gradio as gr
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Treinar um modelo simples
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)
model = RandomForestClassifier()
model.fit(X, y)

def predict(feature1, feature2, feature3, feature4):
    prediction = model.predict([[feature1, feature2, feature3, feature4]])
    return "Classe 1" if prediction[0] == 1 else "Classe 0"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Feature 1"),
        gr.Number(label="Feature 2"),
        gr.Number(label="Feature 3"),
        gr.Number(label="Feature 4")
    ],
    outputs="text",
    title="Classificador"
)
```

### Exercício 3
Crie uma interface para um modelo de regressão que:
1. Aceite múltiplas features numéricas
2. Faça a predição
3. Mostre o resultado com um gráfico de dispersão

## Layouts e Estilização

### Layouts Personalizados
```python
import gradio as gr

def processar_dados(texto, numero):
    return f"Processado: {texto} - {numero}"

with gr.Blocks() as demo:
    gr.Markdown("# Interface Personalizada")
    
    with gr.Row():
        with gr.Column():
            texto = gr.Textbox(label="Texto")
            numero = gr.Number(label="Número")
        with gr.Column():
            output = gr.Textbox(label="Resultado")
    
    btn = gr.Button("Processar")
    btn.click(fn=processar_dados, inputs=[texto, numero], outputs=output)
```

### Estilização
```python
import gradio as gr

css = """
.container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Interface Estilizada")
    # ... componentes ...
```

### Exercício 4
Crie uma interface com:
1. Layout em duas colunas
2. Estilização personalizada
3. Múltiplos componentes interativos
4. Feedback visual para o usuário

## Projeto Final

### Aplicação de Processamento de Imagens

#### Objetivo
Criar uma aplicação completa que:
1. Aceite upload de imagens
2. Aplique diferentes filtros
3. Mostre resultados em tempo real
4. Permita download das imagens processadas

#### Código Base
```python
import gradio as gr
import numpy as np
from PIL import Image, ImageFilter

def aplicar_filtro(imagem, filtro):
    img = Image.fromarray(imagem)
    if filtro == "Blur":
        img = img.filter(ImageFilter.BLUR)
    elif filtro == "Sharpen":
        img = img.filter(ImageFilter.SHARPEN)
    elif filtro == "Edge":
        img = img.filter(ImageFilter.EDGE_ENHANCE)
    return np.array(img)

with gr.Blocks() as demo:
    gr.Markdown("# Processador de Imagens")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Imagem Original")
            filtro = gr.Radio(["Blur", "Sharpen", "Edge"], label="Filtro")
        with gr.Column():
            output_image = gr.Image(label="Imagem Processada")
    
    btn = gr.Button("Processar")
    btn.click(
        fn=aplicar_filtro,
        inputs=[input_image, filtro],
        outputs=output_image
    )
```

### Exercício Final
Desenvolva uma aplicação Gradio para análise de sentimentos que inclua:
1. Input de texto
2. Processamento com modelo de ML
3. Visualização do resultado
4. Interface responsiva e intuitiva
5. Documentação clara

## Recursos Adicionais

### Links Úteis
- [Documentação Oficial do Gradio](https://gradio.app/docs)
- [Gradio Hub](https://huggingface.co/spaces)
- [Exemplos do Gradio](https://gradio.app/gallery)

### Próximos Passos
1. Explorar componentes avançados
2. Implementar autenticação
3. Fazer deploy no Hugging Face Spaces
4. Integrar com APIs externas

---

## Conclusão
Este curso forneceu uma base sólida para começar a desenvolver interfaces interativas com Gradio. Continue praticando e explorando as diversas funcionalidades disponíveis na biblioteca.

Lembre-se: A melhor maneira de aprender é praticando! Crie seus próprios projetos e experimente diferentes recursos do Gradio. 