# Visualização de Dados com Streamlit

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

## Gráficos com Streamlit

Você pode criar **gráficos interativos** com poucas linhas. O Streamlit integra com Pandas e Numpy.

---

### Gráfico de Linha

```python
st.line_chart(dados.set_index('data'))
```

**Esse gráfico mostra a evolução de valores ao longo do tempo.** Ideal para séries temporais.

---

### Gráfico de Barras e Área

```python
st.bar_chart(dados.set_index('data'))
st.area_chart(dados.set_index('data'))
```

O gráfico de barras é ótimo para **comparar categorias**.
O gráfico de área é como o de linha, mas **preenchido**.

---

## Exercício 2

Crie um **mini-dashboard** com:

1. Upload de um arquivo CSV (`st.file_uploader`)
2. Exibir `st.dataframe()` com as primeiras linhas
3. Mostrar `st.line_chart()`
4. Adicionar filtro de data (`st.date_input` ou `query` com Pandas)

---

## Layout e Estilização

### Organização do Layout

Você pode dividir a tela com:

```python
col1, col2 = st.columns(2)
```

Ou agrupar informações em blocos:

```python
with st.expander("Mais detalhes"):
```

E ainda usar a **barra lateral**:

```python
st.sidebar.title("Menu")
```

---

### Estilização

Você pode usar CSS dentro de `st.markdown()` para deixar sua aplicação mais bonita.

---

### Exercício 3

Monte um layout com:

* Menu lateral
* Duas colunas
* Expansores para detalhes
* Cores e estilos personalizados

