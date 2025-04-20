import{_ as o}from"./chunks/colab-badge.CZNv0YKc.js";import{_ as a,c as t,a2 as r,o as n}from"./chunks/framework.H8_ecXae.js";const h=JSON.parse('{"title":"Chatbot ELIZA em Python","description":"","frontmatter":{},"headers":[],"relativePath":"chatbotbook/8_chatbot_eliza_em_python.md","filePath":"chatbotbook/8_chatbot_eliza_em_python.md"}'),s={name:"chatbotbook/8_chatbot_eliza_em_python.md"};function u(d,e,i,m,p,c){return n(),t("div",null,e[0]||(e[0]=[r('<h1 id="chatbot-eliza-em-python" tabindex="-1">Chatbot ELIZA em Python <a class="header-anchor" href="#chatbot-eliza-em-python" aria-label="Permalink to &quot;Chatbot ELIZA em Python&quot;">​</a></h1><h2 id="introducao" tabindex="-1">Introdução <a class="header-anchor" href="#introducao" aria-label="Permalink to &quot;Introdução&quot;">​</a></h2><p>Apresenta-se, nesta seção, uma implementação simplificada em Python de um chatbot inspirado no paradigma ELIZA. Esta implementação demonstra a utilização de expressões regulares para a identificação de padrões textuais (palavras-chave) na entrada fornecida pelo usuário e a subsequente geração de respostas, fundamentada em regras de transformação predefinidas manualmente.</p><p><a href="https://colab.research.google.com/github/giseldo/chatbotbook/blob/main/notebook/eliza.ipynb" target="_blank" rel="noreferrer"><img src="'+o+`" alt="image"></a></p><pre><code>import re  
import random  

regras = [
    (re.compile(r&#39;\\b(hello|hi|hey)\\b&#39;, re.IGNORECASE),
     [&quot;Hello. How do you do. Please tell me your problem.&quot;]),

    (re.compile(r&#39;\\b(I am|I\\&#39;?m) (.+)&#39;, re.IGNORECASE),
     [&quot;How long have you been {1}?&quot;,   
      &quot;Why do you think you are {1}?&quot;]),

    (re.compile(r&#39;\\bI need (.+)&#39;, re.IGNORECASE),
     [&quot;Why do you need {1}?&quot;,
      &quot;Would it really help you to get {1}?&quot;]),

    (re.compile(r&#39;\\bI can\\&#39;?t (.+)&#39;, re.IGNORECASE),
     [&quot;What makes you think you can&#39;t {1}?&quot;,
      &quot;Have you tried {1}?&quot;]),

    (re.compile(r&#39;\\bmy (mother|father|mom|dad)\\b&#39;, re.IGNORECASE),
     [&quot;Tell me more about your family.&quot;,
      &quot;How do you feel about your parents?&quot;]),

    (re.compile(r&#39;\\b(sorry)\\b&#39;, re.IGNORECASE),
     [&quot;Please don&#39;t apologize.&quot;]),

    (re.compile(r&#39;\\b(maybe|perhaps)\\b&#39;, re.IGNORECASE),
     [&quot;You don&#39;t seem certain.&quot;]),

    (re.compile(r&#39;\\bbecause\\b&#39;, re.IGNORECASE),
     [&quot;Is that the real reason?&quot;]),

    (re.compile(r&#39;\\b(are you|do you) (.+)\\?$&#39;, re.IGNORECASE),
     [&quot;Why do you ask that?&quot;]),

    (re.compile(r&#39;\\bcomputer\\b&#39;, re.IGNORECASE),
     [&quot;Do computers worry you?&quot;]),
]

respostas_padrao = [
    &quot;I see.&quot;,  
    &quot;Please tell me more.&quot;,  
    &quot;Can you elaborate on that?&quot;  
]

def response(entrada_usuario):
    for padrao, respostas in regras:
        match = padrao.search(entrada_usuario)  
        if match:
            resposta = random.choice(respostas)
            if match.groups():
                resposta = resposta.format(*match.groups())
            return resposta
    return random.choice(respostas_padrao)


# Exemplo de uso
print(&quot;User: Hello.&quot;)
print(&quot;Bot: &quot; + response(&quot;Hello.&quot;))

print(&quot;User: I am feeling sad.&quot;)
print(&quot;Bot: &quot; + response(&quot;I am feeling sad.&quot;))

print(&quot;Maybe I was not good enough.&quot;)
print(&quot;Bot: &quot; + response(&quot;Maybe I was not good enough.&quot;))

print(&quot;My mother tried to help.&quot;)
print(&quot;Bot: &quot; + response(&quot;My mother tried to help.&quot;))
</code></pre><p>Na implementação, são definidos múltiplos padrões de expressões regulares que correspondem a palavras-chave ou estruturas frasais de interesse (e.g., saudações, construções como &quot;I am&quot; ou &quot;I need&quot;, referências a termos familiares). A função <code>response</code>, ao receber uma string de entrada, itera sequencialmente sobre essas regras. Para cada regra, utiliza-se o método <code>padrao.search(entrada)</code> para verificar a ocorrência do padrão correspondente na sentença do usuário.</p><p>Quando uma correspondência (match) é encontrada, uma resposta associada à regra é selecionada. É relevante notar que diversas respostas predefinidas contêm marcadores (placeholders), como <code>{1}</code>. Estes marcadores são dinamicamente substituídos pelo conteúdo dos grupos capturados pela expressão regular na entrada original do usuário. Por exemplo, se a entrada for <em>&quot;I am sad&quot;</em> e esta corresponder a um padrão como <code>r&quot;I am (.*)&quot;</code>, o grupo capturado (<code>(.*)</code>) conterá a string <em>&quot;sad&quot;</em>. Se a resposta associada for <em>&quot;Why do you think you are {1}?&quot;</em>, o sistema a formatará como <em>&quot;Why do you think you are sad?&quot;</em>, substituindo <code>{1}</code> pelo conteúdo do primeiro grupo capturado.</p><p>Na ausência de correspondência com qualquer regra específica, o sistema seleciona aleatoriamente uma resposta genérica a partir de um conjunto predefinido (denominado, por exemplo, <code>respostas_padrao</code>), tal como <em>&quot;I see.&quot;</em>.</p><h2 id="exemplo-de-interacao" tabindex="-1">Exemplo de Interação <a class="header-anchor" href="#exemplo-de-interacao" aria-label="Permalink to &quot;Exemplo de Interação&quot;">​</a></h2><pre><code>Usuário: Hello
Bot: Hello. How do you do. Please tell me your problem.

Usuário: I am feeling sad
Bot: Why do you think you are feeling sad?

Usuário: Because I lost my job
Bot: Is that the real reason?

Usuário: Maybe I was not good enough.
Bot: You don&#39;t seem certain.

Usuário: My mother tried to help.
Bot: Tell me more about your family.
</code></pre><p>Observa-se na interação que o chatbot inicialmente respondeu à saudação (<em>&quot;Hello...&quot;</em>) com base na regra 1. Em seguida, a entrada <em>&quot;I am sad&quot;</em> ativou a regra 2, resultando em uma resposta que incorpora o termo <em>&quot;sad&quot;</em>. A interação prossegue com o sistema respondendo a entradas iniciadas por <em>&quot;Because...&quot;</em> (regra 8) e <em>&quot;Maybe...&quot;</em> (regra 7), entre outras. Cada resposta gerada foi determinada pela regra correspondente ativada e pelo conteúdo específico da sentença do usuário, mimetizando o comportamento baseado em padrões do sistema ELIZA original [@Abdul-Kader2015].</p>`,11)]))}const b=a(s,[["render",u]]);export{h as __pageData,b as default};
