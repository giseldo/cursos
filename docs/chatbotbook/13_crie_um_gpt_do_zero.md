# Crie um GPT do Zero

## Introdução

Para escrever um **GPT**, precisamos de algumas coisas. Primeiro,
precisamos de um tokenizador. O tokenizador é responsável por dividir o
texto em partes menores (tokens) que o modelo pode entender. Depois,
precisamos de um modelo. O modelo é a parte que realmente faz o trabalho
de entender e gerar texto.

Primeiro, antes de criarmos um tokenizador em Python do Zero, vamos usar
um tokenizador já existente no hugging faces.

[ ![image](./fig/colab-badge.png)
](https://colab.research.google.com/github/giseldo/chatbotbook_v2/blob/main/notebook/cap13.ipynb)

``` {#lst:usando_tokenizer_gpt2 .python language="Python" caption="Usando o tokenizador do GPT-2" label="lst:usando_tokenizer_gpt2"}
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
input= "Olá, como vai você?"    
token_id = tokenizer(input)
print(token_id)
```

    {'input_ids': [30098, 6557, 11, 401, 78, 410, 1872, 12776, 25792, 30], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

A saída deste código será um dicionário com os ids dos tokens e a
máscara de atenção. O id do token é o número que representa cada palavra
ou parte da palavra no vocabulário do modelo. A máscara de atenção
indica quais tokens devem ser considerados pelo modelo durante o
processamento.

Attention mask é uma lista de 1s e 0s que indica quais tokens devem ser
considerados pelo modelo durante o processamento. Um valor de 1
significa que o token correspondente deve ser considerado, enquanto um
valor de 0 significa que ele deve ser ignorado.

[^1]: <https://github.com/keiffster/program-y/wiki/RDF>

[^2]: <https://medium.com/pandorabots-blog/new-feature-visualize-your-aiml-26e33a590da1>

[^3]: <https://www.pandorabots.com/mitsuku/>

[^4]: <https://aisb.org.uk/category/loebner-prize/>
