# Instalação e Configuração

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

### Requisitos
- Python 3.7+
- pip (gerenciador de pacotes Python)

### Instalação
```bash
pip install streamlit
```

### Primeiro App
```python
import streamlit as st

st.title("Meu Primeiro App Streamlit")
st.write("Olá, mundo!")
```

### Executando o App
```bash
streamlit run app.py
```