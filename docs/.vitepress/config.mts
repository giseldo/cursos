import { defineConfig } from 'vitepress'
import { withMermaid } from "vitepress-plugin-mermaid";


// https://vitepress.dev/reference/site-config
export default withMermaid({
  base: '/cursos/',
  lang: 'br',
  title: "Neo Cursos",
  description: "Chatbots com Gradio",
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
      { text: 'Blog', link: 'http://giseldo.github.io/blog' },
    ],

    sidebar: {
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

      '/NLP/': [
        {
          text: 'NLP',
          items: [
            { text: 'Introdução', link: '/nlp/' },
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
      copyright: '© 2024 Giseldo Neo'
    },

    search: {
      provider: 'local'
    }
  }
})


