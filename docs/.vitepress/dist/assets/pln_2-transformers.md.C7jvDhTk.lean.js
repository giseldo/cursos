import{_ as i,c as a,a0 as e,o as t}from"./chunks/framework.B6pkLAvh.js";const E=JSON.parse('{"title":"Transformers","description":"","frontmatter":{},"headers":[],"relativePath":"pln/2-transformers.md","filePath":"pln/2-transformers.md"}'),n={name:"pln/2-transformers.md"};function l(o,s,p,h,r,k){return t(),a("div",null,s[0]||(s[0]=[e(`<h1 id="transformers" tabindex="-1">Transformers <a class="header-anchor" href="#transformers" aria-label="Permalink to &quot;Transformers&quot;">​</a></h1><p>Os modelos transformers são usados para resolver todos os tipos de tarefas de PLN, como algumas mencionadas na seção anterior. Aqui estão algumas empresas e organizações usando a Hugging Face e os modelos Transformers. Estas empresas também contribuem de volta para a comunidade compartilhando seus modelos.</p><ul><li>Facebook AI - 23 modelos</li><li>Microsoft - 33 modelos</li><li>Grammarly - 1 modelo</li><li>Google AI - 115 modelos</li><li>Asteroid-team - 1 modelo</li><li>Allen Institute AI - 43 modelos</li><li>Typerform - modelos</li></ul><p>A biblioteca <a href="https://github.com/huggingface/transformers" target="_blank" rel="noreferrer">Transformers</a> oferece a funcionalidade para criar e usar esses modelos compartilhados. O <a href="https://huggingface.co/models" target="_blank" rel="noreferrer">Model Hub</a> contém milhares de modelos pré-treinados que qualquer um pode baixar e usar. Você pode também dar upload nos seus próprios modelos no Hub!</p><p>O hugging Face Hub não é limitado aos modelos Transformers. Qualquer um pode compartilhar quaisquer tipos de modelos ou datasets que quiserem!</p><p>Exemplos:</p><p>O objeto mais básico na biblioteca Transformers é a função <code>pipeline()</code>. Ela conecta o modelo com seus passos necessários de pré e pós processamento, permitindo-nos diretamente inserir qualquer texto e obter uma resposta.</p><p><a href="https://colab.research.google.com/drive/1BctAyiLiAerEyZxo3JwNqwcBXONv3D_t?usp=sharing" target="_blank" rel="noreferrer">Colab source-code</a></p><div class="language-Python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">Python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">from</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> transformers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pipeline</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pipeline(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;sentiment-analysis&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;I&#39;ve bee waiting for a HuggingFace course my whole life.&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[{</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;label&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;POSITIVE&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;score&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.9516071081161499</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}]</span></span></code></pre></div><p>O modelo cacheado default utilizado foi o <em>distilbert/distilbert-base-uncased-finetuned-sst-2-english</em></p><p>Também podemos passar mais de uma sentença.</p><p><a href="https://colab.research.google.com/drive/1BctAyiLiAerEyZxo3JwNqwcBXONv3D_t?usp=sharing" target="_blank" rel="noreferrer">Colab source-code</a></p><div class="language-Python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">Python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">from</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> transformers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pipeline</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pipeline(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;sentiment-analysis&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier([</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;I&#39;ve bee waiting for a HuggingFace course my whole life.&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;I hate this so much&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span></code></pre></div><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[{</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;label&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;NEGATIVE&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;score&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.6071575880050659</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;label&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;NEGATIVE&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;score&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.9995144605636597</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}]</span></span></code></pre></div><p>Por padrão, esse pipeline seleciona particularmente um modelo pré-treinado que tem sido <em>ajustado</em> (fine-tuned) para análise de sentimentos em inglês. O modelo é baixado e cacheado quando você cria o objeto <code>classifier</code>. Se você rodar novamente o comando, o modelo cacheado será usado e não será baixado novamente.</p><p>Há três principais passos envolvidos quando você passa algum texto para um pipeline:</p><ol><li>O texto é pré-processado para um formato que o modelo consiga entender.</li><li>As entradas (<em>inputs</em>) pré-processados são passadas para o modelo.</li><li>As predições do modelo são pós-processadas, para que então você consiga atribuir sentido a elas.</li></ol><p>Alguns dos pipelines disponíveis atualmente, são:</p><ul><li><em>zero-shot-classification</em> (classificação zero-shot)</li><li><em>text-generation</em> (geração de texto)</li><li><em>feature-extraction</em> (pega a representação vetorial do texto)</li><li><em>fill-mask</em> (preenchimento de máscara)</li><li><em>ner</em> (reconhecimento de entidades nomeadas)</li><li><em>question-answering</em> (responder perguntas)</li><li><em>sentiment-analysis</em> (análise de sentimentos)</li><li><em>summarization</em> (sumarização)</li><li><em>translation</em> (tradução)</li></ul><h2 id="classificacao-zero-shot" tabindex="-1">Classificação Zero-shot <a class="header-anchor" href="#classificacao-zero-shot" aria-label="Permalink to &quot;Classificação Zero-shot&quot;">​</a></h2><p>Esse é um cenário comum nos projetos reais porque anotar texto geralmente consome bastante tempo e requer expertize no domínio. Para esse caso, o pipeline <code>zero-sho-classification</code> é muito poderoso: permite você especificar quais os rótulos usar para a classificação, desse modo você não precisa &quot;confiar&quot; nos rótulos pré-treinados. Você já viu como um modelo pode classificar uma sentença como positiva ou negativa usando esses dois rótulos - mas também pode ser classificado usando qualquer outro conjunto de rótulos que você quiser.</p><p><a href="https://colab.research.google.com/drive/1BctAyiLiAerEyZxo3JwNqwcBXONv3D_t?usp=sharing" target="_blank" rel="noreferrer">Colab source-code</a></p><div class="language-Python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">Python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">from</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> transformers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pipeline</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pipeline(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;zero-shot-classification&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;This is a course about the Transformers library&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">          candidate_labels</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;education&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;politics&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;business&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span></code></pre></div><p>O modelo padrão utilizado foi <em>facebook/bart-large-mnli</em></p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">{</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">&#39;sequence&#39;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &#39;This is a course about the Transformers library&#39;,</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> &#39;labels&#39;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;education&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;business&#39;,</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &#39;politics&#39;],</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> &#39;scores&#39;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [0.8445994257926941, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">0.11197380721569061,</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> 0.04342673346400261]}</span></span></code></pre></div><p>Esse pipeline é chamado de <em>zero-shot</em> porque você não precisa fazer o ajuste fino do modelo nos dados que você o utiliza.</p><h2 id="geracao-de-texto" tabindex="-1">Geração de Texto <a class="header-anchor" href="#geracao-de-texto" aria-label="Permalink to &quot;Geração de Texto&quot;">​</a></h2><p>A principal ideia aqui é que você coloque um pedaço de texto e o modelo irá autocompletá-lo ao gerar o texto restante. Isso é similar ao recurso de predição textual que é encontrado em inúmeros celulares. A geração de texto envolve aleatoriedade, então é normal se você não obter o mesmo resultado obtido mostrado abaixo.</p><p><a href="https://colab.research.google.com/drive/1BctAyiLiAerEyZxo3JwNqwcBXONv3D_t?usp=sharing" target="_blank" rel="noreferrer">Colab source-code</a></p><div class="language-Python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">Python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">from</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> transformers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pipeline</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">generator </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pipeline(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;text-generation&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">generator(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;In this course, we will teach you how to&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>O modelo padrão utilizado foi o <em>openai-community/gpt2</em>.</p><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[{</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;generated_text&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;&quot;&quot;In this course, we will teach you how to  navigate the real world using the virtual world which can serve  as a powerful tool to help you develop skills in the real world  and learn skills in the virtual world.\\n\\nFor the VirtualWorld course,&quot;&quot;&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}]</span></span></code></pre></div><p>Você pode controlar quão diferentes sequências são geradas com o argumento <code>num_return_sequences</code> e o tamanho total da saída de texto (<em>output</em>) com o argumento <code>max_length</code>.</p><p><a href="https://colab.research.google.com/drive/1BctAyiLiAerEyZxo3JwNqwcBXONv3D_t?usp=sharing" target="_blank" rel="noreferrer">Colab source-code</a></p><div class="language-Python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">Python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">from</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> transformers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pipeline</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">generator </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pipeline(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;text-generation&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">generator(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;In this course, we will teach you how to&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">num_return_sequences</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">max_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>O modelo padrão utilizado foi o <em>openai-community/gpt2</em>.</p><div class="language-Python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">Python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[{</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;generated_text&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;&quot;&quot;In this course, we will teach you how to develop and use your voice to help others around you understand how you can help them.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">The&quot;&quot;&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;generated_text&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;&quot;&quot;In this course, we will teach you how to  create multiple different web applications to run in multiple  languages, providing you a complete framework for writing an  application&quot;&quot;&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}]</span></span></code></pre></div><h3 id="usando-qualquer-modelo-do-hub-em-um-pipeline" tabindex="-1">Usando qualquer modelo do Hub em um pipeline <a class="header-anchor" href="#usando-qualquer-modelo-do-hub-em-um-pipeline" aria-label="Permalink to &quot;Usando qualquer modelo do Hub em um pipeline&quot;">​</a></h3><p>Nos exemplos passados, usamos o modelo padrão para a tarefa que executamos, mas você pode usar um modelo particular do Hub para usá-lo no pipeline em uma tarefa específica, tal como, geração de texto (<em>text-generation</em>). Vá ao <a href="https://huggingface.co/models" target="_blank" rel="noreferrer">Model Hub</a> e clique <em>Edit filters</em>, selecione <em>text_generation</em> na esquerda para mostrar apenas os modelos suportáveis para esta tarefa.</p><p>Vamos utilizar o modelo <em>distilgpt2</em>.</p><p><a href="https://colab.research.google.com/drive/1BctAyiLiAerEyZxo3JwNqwcBXONv3D_t?usp=sharing" target="_blank" rel="noreferrer">Colab source-code</a></p><div class="language-Python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">Python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">from</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> transformers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pipeline</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">generator </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pipeline(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;text-generation&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">model</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;distilgpt2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">generator (</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;In this course, we will teach you how to&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">          max_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">30</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">          num_return_sequences</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language-shell vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">shell</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[{</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;generated_text&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;&quot;&quot;In this course, we will teach you how to write a new language without using anything new. For example; as it is written, the same language&quot;&quot;&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&#39;generated_text&#39;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;&quot;&quot;In this course, we will teach you how to  solve this problem through a real, real, and real data-centric approach: an algorithm that combines&quot;&quot;&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}]</span></span></code></pre></div><p>Experimente! Use os filtros para encontrar um modelo de geração de texto em outra lingua no <a href="https://huggingface.co/models" target="_blank" rel="noreferrer">Model Hub</a>. Veja na figura à seguir.</p><p>Uma vez que você seleciona o modelo clicando nele, você irá ver que há um widget que permite que você teste-o diretamente online. Desse modo você pode rapidamente testar as capacidades do modelo antes de baixa-lo. Veja na figura à seguir.</p>`,46)]))}const c=i(n,[["render",l]]);export{E as __pageData,c as default};