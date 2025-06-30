# **Atividade: Criando seu Primeiro Aplicativo com Streamlit**

:::info
Criado por Alana Neo.
:::

## **Objetivo:**

Aprender a criar um pequeno aplicativo interativo usando Python e Streamlit.

---

## **Criando o aplicativo**

Copie e cole este código dentro do arquivo:

```python
import streamlit as st

st.title("Minha Página Interativa")
st.header("Bem-vindo ao meu primeiro app!")

nome = st.text_input("Qual é o seu nome?")
idade = st.number_input("Quantos anos você tem?", min_value=1, max_value=120)
cor = st.selectbox("Qual sua cor favorita?", ["Vermelho", "Verde", "Azul", "Amarelo"])

if st.button("Enviar"):
    st.write("Olá,", nome + "!")
    st.write("Você tem", idade, "anos.")
    st.write("Sua cor favorita é:", cor)

st.write("Obrigado por usar meu app.")
```

## **Desafio**

Agora, edite o código e **adicione mais dois componentes** abaixo:

* Um **checkbox** para perguntar se a pessoa gosta de tecnologia
* Um **slider** para perguntar de 0 a 10 quanto ela gosta de matemática

**Dica**:

```python
tecnologia = st.checkbox("Você gosta de tecnologia?")
nota_matematica = st.slider("Quanto você gosta de matemática?", 0, 10)
```

---

## **Resultado Esperado:**

O aluno deve conseguir visualizar uma página com os seguintes itens:

* Título e cabeçalho
* Caixa para digitar nome
* Caixa para escolher idade
* Caixa de seleção com cores
* Botão para enviar as informações
* Textos que mostram as escolhas
* Checkbox e slider (caso faça o desafio)