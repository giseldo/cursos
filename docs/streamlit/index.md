# **Tutorial de Streamlit**

:::info
Criado por Alana Neo.
:::

## **O que é o Streamlit?**

O **Streamlit** é uma framework que permite criar páginas interativas usando a linguagem Python. Ele é usado para fazer aplicativos simples que mostram gráficos, textos, imagens e botões.

---

## **Criando o primeiro arquivo**

Esse será o nosso programa principal. No editor de texto, copie o código abaixo:

```python
import streamlit as st

st.title("Meu Primeiro App com Streamlit")
st.write("Olá! Este é um aplicativo feito com Streamlit.")
```

Atualize a página.

---

## **Componentes Básicos do Streamlit**

Abaixo estão os principais elementos (componentes) que podemos usar no Streamlit:

---

### **1. Título e Texto**

Usamos esses para mostrar informações na tela.

```python
st.title("Título Principal")
st.header("Cabeçalho")
st.subheader("Subcabeçalho")
st.write("Este é um texto comum.")
```

---

### **2. Imagens**

Você pode mostrar uma imagem da internet ou do seu computador.

```python
st.image("https://via.placeholder.com/150", caption="Exemplo de imagem")
```

---

### **3. Botões**

Permitem que o usuário clique e aconteça alguma coisa.

```python
if st.button("Clique aqui"):
    st.write("Você clicou no botão!")
```

---

### **4. Caixa de Texto**

Para o usuário digitar alguma coisa.

```python
nome = st.text_input("Digite seu nome:")
st.write("Você digitou:", nome)
```

---

### **5. Caixa de Número**

O usuário escolhe um número.

```python
idade = st.number_input("Digite sua idade:", min_value=1, max_value=120)
st.write("Idade:", idade)
```

---

### **6. Caixa de Seleção (Caixa de Opções)**

Permite escolher uma opção de uma lista.

```python
cor = st.selectbox("Escolha uma cor:", ["Vermelho", "Verde", "Azul"])
st.write("Você escolheu:", cor)
```

---

### **7. Caixa de Verificação**

Para marcar ou desmarcar opções.

```python
aceita = st.checkbox("Aceito os termos")
if aceita:
    st.write("Obrigado!")
```

---

### **8. Barras Deslizantes**

Permite escolher valores arrastando uma barra.

```python
nota = st.slider("Escolha uma nota de 0 a 10", 0, 10)
st.write("Nota escolhida:", nota)
```

---

### **9. Gráficos Simples**

Você pode mostrar gráficos com poucas linhas de código.

```python
import pandas as pd
import matplotlib.pyplot as plt

dados = pd.DataFrame({
    'Nomes': ['A', 'B', 'C'],
    'Notas': [8, 6, 9]
})

st.bar_chart(dados.set_index('Nomes'))
```

---

## **Resumo**

| Componente        | Código            | Serve para...              |
| ----------------- | ----------------- | -------------------------- |
| `st.title()`      | Título            | Mostrar um título grande   |
| `st.write()`      | Texto             | Mostrar texto na tela      |
| `st.button()`     | Botão             | Fazer uma ação com clique  |
| `st.text_input()` | Entrada de texto  | O usuário digitar algo     |
| `st.selectbox()`  | Lista de opções   | Escolher uma entre várias  |
| `st.checkbox()`   | Marcar opção      | Ativar ou desativar algo   |
| `st.slider()`     | Barra de valor    | Escolher um valor numérico |
| `st.bar_chart()`  | Gráfico de barras | Mostrar gráficos           |

---

## **Conclusão**

Com o Streamlit, você pode transformar seus programas Python em páginas interativas e fáceis de usar. Ele é ótimo para quem está começando e quer ver seus códigos funcionando de forma visual.
