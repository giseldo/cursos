import{_ as t,c as a,j as e,a as o,o as n}from"./chunks/framework.H8_ecXae.js";const l=JSON.parse('{"title":"Crie um GPT do Zero","description":"","frontmatter":{},"headers":[],"relativePath":"chatbotbook/13_crie_um_gpt_do_zero.md","filePath":"chatbotbook/13_crie_um_gpt_do_zero.md"}'),i={name:"chatbotbook/13_crie_um_gpt_do_zero.md"};function s(c,r,m,_,d,p){return n(),a("div",null,r[0]||(r[0]=[e("h1",{id:"crie-um-gpt-do-zero",tabindex:"-1"},[o("Crie um GPT do Zero "),e("a",{class:"header-anchor",href:"#crie-um-gpt-do-zero","aria-label":'Permalink to "Crie um GPT do Zero"'},"​")],-1),e("p",null,[o("Para escrever um "),e("strong",null,"GPT"),o(", precisamos de algumas coisas. Primeiro vamos criar um tokenizador em Python. Mas para facilitar, vamos usar um tokenizador já existente no hugging faces.")],-1),e("pre",null,[e("code",null,`from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input= "Olá, como vai você?"    

token_id = tokenizador(input)

print(token_id)
`)],-1)]))}const f=t(i,[["render",s]]);export{l as __pageData,f as default};
