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
      { text: 'Sobre', link: '/pages/sobre' }
    ],

    sidebar: {

      '/chatbotbook/': [
        {
          text: 'Chatbotbook',
          items: [
            { text: 'Index', link: '/chatbotbook/' },
            { text: 'Definições e contexto', link: '/chatbotbook/1_chatbots_definições_e_contexto' },
            { text: 'Eliza Explicado', link: '/chatbotbook/2_eliza_explicado' },
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
          items: [
            { text: 'Pag 1', link: '/estatistica/' },
            { text: 'Pag 2', link: '/estatistica/pag2' }                        
          ]
        }
      ],

      '/am/': [
        {
          text: 'AM',
          items: [
            { text: 'Introdução', link: '/am/' },
            { text: 'Pag 2', link: '/am/cap2' },
            { text: 'Pag 3', link: '/am/cap3' },
            { text: 'Pag 4', link: '/am/cap4' },
            { text: 'Pag 4', link: '/am/cap5' }                        
          ]
        }
      ],

      '/pln/': [
        {
          text: 'Processamento de linguagem natural',
          items: [
            { text: 'Sobre', link: '/pln/' },
            { text: 'Introdução', link: '/pln/1-introducao' },
            { text: 'Transformers', link: '/pln/2-transformers' }                        
          ]
        }
      ],

      '/nlp/': [
        {
          text: 'NLP',
          items: [
            { text: 'Introdução', link: '/nlp/' },
            ]
        }
      ],

      '/llm/': [
        {
          text: 'LLM (medium)',
          items: [
            { text: 'Large Language Models: A Short Introduction', link: '/llm/' },
            { text: 'A Very Gentle Introduction to Large Language Models without the Hype', link: '/llm/verygentle' },
            
          ]
        }
      ],

      '/transformers/': [
        {
          text: 'Transformers',
          items: [
            { text: 'Introdução', link: '/transformers/' },
          ]
        }
      ],

      '/chatbot/': [
        {
          text: 'Chatbot',
          items: [
            { text: 'Sobre', link: '/chatbot/' },
            { text: 'Como criar um chatbot', link: '/chatbot/1-como-criar' }
          ]
        }
      ],

      '/regressao/': [
        {
          text: 'Regressao Linear',
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
  }
})


