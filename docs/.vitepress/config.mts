import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/cursos/',
  lang: 'br',
  title: "Neo Cursos",
  description: "Chatbots com Gradio",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: 
    [
      { text: 'Home', link: '/' },
      { text: 'Sobre', link: '/pages/sobre' },
      { text: 'Blog', link: 'http://giseldo.github.io/blog' },
    ],

    sidebar: {
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
      { icon: 'github', link: 'https://github.com/giseldo' }
    ],

    footer: {
      message: 'Lançado sob a Licença MIT.',
      copyright: 'Direitos autorais © 2024 Giseldo Neo'
    },

    search: {
      provider: 'local'
    }
  }
})


