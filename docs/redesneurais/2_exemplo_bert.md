# Exemplo de Uso do BERT com Hugging Face

Este exemplo demonstra como usar o modelo pré-treinado BERT para realizar tarefas de NLP, como classificação de texto.

## Instalação
Certifique-se de instalar a biblioteca `transformers`:
```bash
pip install transformers torch
```

```python
from transformers import pipeline

clf = pipeline('fill-mask', model="bert-base-uncased")

response = clf("The book is on the [MASK].")

response

```



```python
from transformers import pipeline

clf = pipeline('text-generation', model="gpt2");
response = clf("The book is on");

print(response[0]['generated_text'])
```

```
The book is on the subject of "The End of the World," a political and economic message that is being trumpeted by many right-wing think tanks and corporate interests. According to the website of the University of Michigan, the book is "a critical review of the Reagan-Bush era agenda."

In a 2012 conference call with reporters, George W. Bush said, "I just want to make it very clear that I don't believe we're in the end of the world. This is just the beginning. The world is not going to be going well, and we're not going to be going very well."

The book was one of dozens of books by liberal activists who are trying to end the disastrous and unjust war on Iraq.

Some of the authors of the book, including the former head of the CIA's Gulf War effort, John Lewis, have also taken on the cause of U.S. intervention.

The book, in a broadside against President Bush, is being billed as a "new approach" to the war on Iraq.

The author, Lewis said, is "not in a position to dictate how the war will be waged, but to set the stage for the next phase of the war."
```

## Código de Exemplo
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Carregar o tokenizer e o modelo pré-treinado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Texto de entrada
text = "Hugging Face está revolucionando o NLP!"

# Tokenizar o texto
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Fazer a inferência
outputs = model(**inputs)
logits = outputs.logits

# Obter a previsão
predicted_class = torch.argmax(logits, dim=1).item()
print(f"Classe prevista: {predicted_class}")
```

## Explicação
1. **Tokenizer**: Converte o texto em tokens que o modelo pode entender.
2. **Modelo**: `BertForSequenceClassification` é usado para classificação de texto.
3. **Inferência**: O modelo retorna logits, que são convertidos em classes previstas.

## Referências
- [Documentação do Hugging Face](https://huggingface.co/docs/transformers)
- [Modelos BERT](https://huggingface.co/bert-base-uncased)