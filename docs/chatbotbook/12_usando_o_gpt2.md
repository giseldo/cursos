# Usando o GPT2

A biblioteca transformers da Hugging Face torna muito mais fácil
trabalhar com modelos pré-treinados como GPT-2. Aqui está um exemplo de
como gerar texto usando o GPT-2 pré-treinado:

``` {#lst:gpt2_exemplo .python language="Python" caption="Exemplo de uso do GPT-2 com a biblioteca transformers" label="lst:gpt2_exemplo"}
from transformers import pipeline

pipe = pipeline('text-generation', model='gpt2')

input = 'Olá, como vai você?'

output = pipe(input)

print(output[0]['generated_text'])
```

Este código é simples porque ele usa um modelo que já foi treinado em um
grande dataset. Também é possível ajustar (fine-tune) um modelo
pré-treinado em seus próprios dados para obter resultados melhores.

[^1]: <https://github.com/keiffster/program-y/wiki/RDF>

[^2]: <https://medium.com/pandorabots-blog/new-feature-visualize-your-aiml-26e33a590da1>

[^3]: <https://www.pandorabots.com/mitsuku/>

[^4]: <https://aisb.org.uk/category/loebner-prize/>
