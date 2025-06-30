# Curso Completo de Streamlit

:::info
Criado por Alana Neo.
:::

:::info
<div style="display: flex; gap: 10px; align-items: center;">
  <img src="./fig/esquerda.jpg" style="width:5%" alt="image" />
  <img src="./fig/direita.jpg" style="width:5%" alt="image" />
  Você pode usar as setas do teclado para navegar entre as seções.
</div>
:::

## Introdução ao Streamlit

### O que é Streamlit?
Streamlit é uma biblioteca Python open-source que permite criar aplicações web interativas para ciência de dados e machine learning de forma rápida e simples. Com Streamlit, você pode transformar scripts Python em aplicações web interativas em minutos.

### Por que usar Streamlit?
- Fácil de aprender e usar
- Não requer conhecimento de frontend
- Integração perfeita com bibliotecas de data science
- Deploy simples
- Comunidade ativa

## Componentes Básicos



### Widgets Interativos

#### Texto e Títulos
```python
st.title("Título Principal")
st.header("Cabeçalho")
st.subheader("Subcabeçalho")
st.text("Texto simples")
st.markdown("**Texto em Markdown**")
```

#### Inputs
```python
# Texto
nome = st.text_input("Digite seu nome")

# Número
idade = st.number_input("Digite sua idade", min_value=0, max_value=120)

# Slider
valor = st.slider("Selecione um valor", 0, 100)

# Checkbox
aceito = st.checkbox("Eu aceito os termos")

# Selectbox
opcao = st.selectbox("Escolha uma opção", ["Opção 1", "Opção 2", "Opção 3"])
```

### Exercício 1
Crie um formulário de cadastro com os seguintes campos:
- Nome
- Email
- Idade
- Gênero (selectbox)
- Termos de uso (checkbox)

## Visualização de Dados

### Gráficos com Streamlit

#### Gráfico de Linha
```python
import pandas as pd
import numpy as np

# Dados de exemplo
dados = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=10),
    'valor': np.random.randn(10).cumsum()
})

st.line_chart(dados.set_index('data'))
```

#### Gráfico de Barras
```python
st.bar_chart(dados.set_index('data'))
```

#### Gráfico de Área
```python
st.area_chart(dados.set_index('data'))
```

### Exercício 2
Crie um dashboard que:
1. Carregue um arquivo CSV
2. Mostre as primeiras linhas do DataFrame
3. Exiba um gráfico de linha com os dados
4. Adicione um filtro por data

## Layout e Estilização

### Organização do Layout
```python
# Colunas
col1, col2 = st.columns(2)
with col1:
    st.write("Coluna 1")
with col2:
    st.write("Coluna 2")

# Expansores
with st.expander("Clique para expandir"):
    st.write("Conteúdo expandido")

# Sidebar
st.sidebar.title("Menu Lateral")
```

### Estilização
```python
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
```

### Exercício 3
Crie um dashboard com:
1. Menu lateral com opções
2. Duas colunas principais
3. Expansores para informações detalhadas
4. Estilização personalizada

## Projeto Final

### Aplicação de Análise de Dados

#### Objetivo
Criar uma aplicação completa que:
1. Carregue dados de um arquivo CSV
2. Permita filtros interativos
3. Mostre diferentes visualizações
4. Inclua métricas importantes
5. Tenha um layout profissional

#### Código Base
```python
import streamlit as st
import pandas as pd
import plotly.express as px

# Configuração da página
st.set_page_config(page_title="Dashboard de Análise", layout="wide")

# Título
st.title("Dashboard de Análise de Dados")

# Carregamento de dados
@st.cache_data
def load_data():
    return pd.read_csv("dados.csv")

# Interface principal
df = load_data()

# Filtros
st.sidebar.header("Filtros")
filtro_categoria = st.sidebar.multiselect("Categoria", df['categoria'].unique())

# Visualizações
col1, col2 = st.columns(2)

with col1:
    st.subheader("Gráfico de Barras")
    fig = px.bar(df, x='categoria', y='valor')
    st.plotly_chart(fig)

with col2:
    st.subheader("Gráfico de Pizza")
    fig = px.pie(df, values='valor', names='categoria')
    st.plotly_chart(fig)

# Métricas
st.subheader("Métricas")
col1, col2, col3 = st.columns(3)
col1.metric("Total", df['valor'].sum())
col2.metric("Média", df['valor'].mean())
col3.metric("Máximo", df['valor'].max())
```

### Exercício Final
Desenvolva uma aplicação Streamlit para análise de dados de vendas que inclua:
1. Carregamento de dados
2. Filtros interativos
3. Múltiplas visualizações
4. Métricas importantes
5. Layout responsivo
6. Documentação clara

## Recursos Adicionais

### Links Úteis
- [Documentação Oficial do Streamlit](https://docs.streamlit.io)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Components](https://streamlit.io/components)

### Próximos Passos
1. Aprender sobre cache e performance
2. Explorar componentes personalizados
3. Implementar autenticação
4. Fazer deploy da aplicação

---

## Conclusão
Este curso forneceu uma base sólida para começar a desenvolver aplicações com Streamlit. Continue praticando e explorando as diversas funcionalidades disponíveis na biblioteca.

Lembre-se: A melhor maneira de aprender é praticando! Crie seus próprios projetos e experimente diferentes recursos do Streamlit. 