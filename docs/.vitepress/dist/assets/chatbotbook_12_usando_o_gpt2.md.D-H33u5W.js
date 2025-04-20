import{_ as s,c as n,a2 as e,o}from"./chunks/framework.H8_ecXae.js";const m=JSON.parse('{"title":"Usando o GPT2","description":"","frontmatter":{},"headers":[],"relativePath":"chatbotbook/12_usando_o_gpt2.md","filePath":"chatbotbook/12_usando_o_gpt2.md"}'),t={name:"chatbotbook/12_usando_o_gpt2.md"};function p(i,a,r,l,d,c){return o(),n("div",null,a[0]||(a[0]=[e(`<h1 id="usando-o-gpt2" tabindex="-1">Usando o GPT2 <a class="header-anchor" href="#usando-o-gpt2" aria-label="Permalink to &quot;Usando o GPT2&quot;">​</a></h1><p>A biblioteca transformers da Hugging Face torna muito mais fácil trabalhar com modelos pré-treinados como GPT-2. Aqui está um exemplo de como gerar texto usando o GPT-2 pré-treinado:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>from transformers import pipeline</span></span>
<span class="line"><span></span></span>
<span class="line"><span>pipe = pipeline(&#39;text-generation&#39;, model=&#39;gpt2&#39;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>input = &#39;Olá, como vai você?&#39;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>output = pipe(input)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>print(output[0][&#39;generated_text&#39;])</span></span></code></pre></div><p>Este código é simples porque ele usa um modelo que já foi treinado em um grande dataset. Também é possível ajustar (fine-tune) um modelo pré-treinado em seus próprios dados para obter resultados melhores.</p>`,4)]))}const _=s(t,[["render",p]]);export{m as __pageData,_ as default};
