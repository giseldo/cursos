# Crie um GPT do Zero

## Introdução

Para escrever um **GPT**, precisamos de algumas coisas. Primeiro vamos
criar um tokenizador em Python. Mas para facilitar, vamos usar um
tokenizador já existente no hugging faces.

    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    input= "Olá, como vai você?"    

    token_id = tokenizador(input)

    print(token_id)

[^1]: <https://github.com/keiffster/program-y/wiki/RDF>

[^2]: <https://medium.com/pandorabots-blog/new-feature-visualize-your-aiml-26e33a590da1>

[^3]: <https://www.pandorabots.com/mitsuku/>

[^4]: <https://aisb.org.uk/category/loebner-prize/>
