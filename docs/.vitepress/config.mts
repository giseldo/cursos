import { defineConfig } from 'vitepress'
import { withMermaid } from "vitepress-plugin-mermaid";


// https://vitepress.dev/reference/site-config
export default withMermaid({
  base: '/cursos/',
  lang: 'br',
  title: "Neo",
  description: "Cursos",
  markdown: {
    math: true
  },
  mermaid: {
    // refer https://mermaid.js.org/config/setup/modules/mermaidAPI.html#mermaidapi-configuration-defaults for options
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    outline: {
      level: [2, 3] // Define o nível global do TOC
    },
    
    nav: 
    [
      { text: 'Home', link: '/' },
      { text: 'Sobre', link: '/pages/sobre' },
      { text: 'Site Principal', link: 'http://giseldo.github.io' }
    ],

    sidebar: {

        '/nodejs/': [
          {
            text: 'Node JS',
            collapsible: true,
            items: [
              { text: 'Introdução ao Node.js', link: '/nodejs/' },
              { text: 'Módulos e CommonJS vs. ES Modules', link: '/nodejs/2. CommonJS_vs_ES_Modules' },
              { text: 'Gerenciamento de Pacotes com npm/Yarn', link: '/nodejs/3. Gerenciamento_de_Pacotes_npm_Yarn' },
              { text: 'Projeto Prático - Servidor HTTP Simples', link: '/nodejs/4. Servidor_HTTP_Simples' },
            ]
          }
        ],

        '/redesneurais/': [
          {
            text: 'Redes Neurais',
            collapsible: true,
            items: [
              { text: 'Inicio', link: '/redesneurais/' },
              { text: 'Bert vs GPT', link: '/redesneurais/1_bert_vs_gpt' },
              { text: 'GPT', link: '/redesneurais/2_gpt' },
              { text: 'BERT', link: '/redesneurais/3_bert' },
            ]
          }
        ],

        '/llmdozero/': [
          {
            text: 'LLM do Zero',
            collapsible: true,
            items: [
              { text: 'Início', link: '/llmdozero/' },  
              { text: 'Introdução', link: '/llmdozero/md1_introducao' },  
              { text: 'O que são LLM', link: '/llmdozero/md1_aula1_o_que_sao_llms' },
              { text: 'Requisitos Hardware', link: '/llmdozero/md1_aula2_requisitos_hardware' },
              { text: 'Fundamentos Matemáticos', link: '/llmdozero/md1_aula3_fundamentos_matematicos' },
              { text: 'Configuracao do ambiente', link: '/llmdozero/md2_aula1_configuracao_ambiente' },
              { text: 'Mnipulacao dados', link: '/llmdozero/md2_aula2_manipulacao_dados' },
              { text: 'Componentes transformers', link: '/llmdozero/md3_aula1_componentes_transformer' },
              { text: 'LLm compacto', link: '/llmdozero/md4_aula1_llm_compacto' },
              { text: 'Treinamento modelo', link: '/llmdozero/md5_aula1_treinamento_modelo' },
              { text: 'Fine Tuning', link: '/llmdozero/md6_aula1_fine_tuning' },
              { text: 'Otimização inferência', link: '/llmdozero/md7_aula1_otimizacao_inferencia' },
              { text: 'Projetos práticos', link: '/llmdozero/md8_aula1_projetos_praticos' }
            ]
          }
        ],

       '/frameworks/': [
        {
          text: 'Frameworks',
          collapsible: true,
          items: [
            { text: 'Início', link: '/frameworks/' },
            { text: 'Mapeamento Objeto Relacional', link: '/frameworks/1-objeto-relacional' },
            { text: 'Persistencia', link: '/frameworks/2-persistencia' },
            { text: 'Padrões', link: '/frameworks/3-padroes' },
            { text: 'MVC', link: '/frameworks/4-MVC' },
            { text: 'Frameworks', link: '/frameworks/5-Frameworks' },
            { text: 'REST', link: '/frameworks/6-REST' }
          ]
        }
      ],

      '/streamlit/': [
        {
          text: 'Streamlit',
          collapsible: true,
          items: [
            { text: 'Aula 1', link: '/streamlit/' },
            { text: 'Aula 1 atividades', link: '/streamlit/atividades' },
          ]
        }
      ],

      '/gradio/': [
        {
          text: 'Gradio',
          collapsible: true,
          items: [
            { text: 'Início', link: '/gradio/' }
          ]
        }
      ],

      '/python/': [
        {
          text: 'Python',
          collapsible: true,
          items: [
            { text: 'Início', link: '/python/' },
            { text: 'Introdução', link: '/python/1-introducao' },
            { text: 'Esturuturas de controle', link: '/python/2-estruturas_controle' },
            { text: 'Funções e módulos', link: '/python/3-funcoes_modulos' },
            { text: 'Estrutura de dados', link: '/python/4-estruturas_dados' },
            { text: 'Manipulação de arquivos', link: '/python/5-manipulacao_arquivos' }
          ]
        }
      ],

      '/chatbotbook/': [
        {
          text: 'Chatbot',
          collapsible: true,
          items: [
            { text: 'Index', link: '/chatbotbook/' },
            { text: 'Definições e contexto', link: '/chatbotbook/1_chatbots_definições_e_contexto' },
            { text: 'ELIZA o primeiro chatbot', link: '/chatbotbook/2_eliza_o_primeiro_chatbot' },
            { text: 'Artificial Intelligence Markup Language', link: '/chatbotbook/3_artificial_intelligence_markup_language' },
            { text: 'Processamento de Linguagem Natural', link: '/chatbotbook/4_processamento_de_linguagem_natural' },
            { text: 'Intenção em chatbots', link: '/chatbotbook/5_intenção_em_chatbots_chap_intents_' },
            { text: 'Large Language Models', link: '/chatbotbook/6_llm' },
            { text: 'Retrieval Augmented Generation', link: '/chatbotbook/7_retrieval_augmented_generation' },
            { text: 'Chatbot Eliza em Python', link: '/chatbotbook/8_chatbot_eliza_em_python' },
            { text: 'Usando Chatgpt com Langchain', link: '/chatbotbook/9_usando_chatgpt_com_langchain' },
            { text: 'Criando Chatbots com LLMs através da Engenharia de Prompts', link: '/chatbotbook/10_criando_chatbots_com_llms_através_da_engenharia_de_prompts' },
            { text: 'Expressões Regulares', link: '/chatbotbook/11_expressões_regulares' },
            { text: 'Usando o Gpt2', link: '/chatbotbook/12_usando_o_gpt2' },
            { text: 'Crie um GPT do Zero', link: '/chatbotbook/13_crie_um_gpt_do_zero.md' },
            
          ]
        }
      ],
      '/estatistica/': [
        {
          text: 'Estatística',
          collapsible: true,
          items: [
            { text: 'Início', link: '/estatistica/' }
          ]
        }
      ],

      '/am/': [
        {
          text: 'AM',
          collapsible: true,
          items: [
            { text: 'Introdução', link: '/am/' },
            { text: 'Preparação dos dados', link: '/am/1_preparacao' },
            { text: 'Inteligência Artificial', link: '/am/2_ia' },
            { text: 'Baseado em árvore', link: '/am/3_baseadoarvore' },
            
          ]
        }
      ],

      '/pln/': [
        {
          text: 'Processamento de linguagem natural',
          collapsible: true,
          items: [
            { text: 'Sobre', link: '/pln/' },
            { text: 'Introdução', link: '/pln/1-introducao' },
            { text: 'Transformers', link: '/pln/2-transformers' },                        
            { text: 'Arquitetura', link: '/pln/3-arquitetura' }                        
          ]
        }
      ],

      '/nlp/': [
        {
          text: 'NLP',
          collapsible: true,
          items: [
            { text: 'Introdução', link: '/nlp/' },
            ]
        }
      ],

      '/llm/': [
        {
          text: 'LLM (medium)',
          collapsible: true,
          items: [
            { text: 'Large Language Models: A Short Introduction', link: '/llm/' },
            { text: 'A Very Gentle Introduction to Large Language Models without the Hype', link: '/llm/verygentle' },
            
          ]
        }
      ],

      '/transformers/': [
        {
          text: 'Transformers',
          collapsible: true,
          items: [
            { text: 'Introdução', link: '/transformers/' },
          ]
        }
      ],

      '/chatbot/': [
        {
          text: 'Chatbot',
          collapsible: true,
          items: [
            { text: 'Início', link: '/chatbot/' },
            { text: 'Gradio', link: '/chatbot/gradio' }
          ]
        }
      ],

      '/regressao/': [
        {
          text: 'Regressao Linear',
          collapsible: true,
          items: [
            { text: 'Sobre', link: '/regressao/' },
            { text: 'Introdução', link: '/regressao/1-intro' },
            { text: 'Vídeo', link: '/regressao/2-video' },
            { text: 'Exercício', link: '/regressao/3-exercicios' },
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/giseldo/curdos' },
      { icon: 'twitter', link: 'https://twitter.com/giseldoneo' },
      { icon: 'instagram', link: 'https://instagram.com/neogiseldo' },
      { icon: 'youtube', link: 'https://youtube.com/giseldoneo' }
    ],

    footer: {
      message: 'Todos os direitos reservados.',
      copyright: '© 2025 Giseldo Neo'
    },

    search: {
      provider: 'local'
    }
  },
  head: [
    ['script', { src: '/login.js' }]
  ]
})
