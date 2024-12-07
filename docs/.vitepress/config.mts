import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/cursos/',
  lang: 'br',
  title: "Neo Cursos",
  description: "Chatbots com Gradio",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Chatbot', link: '/chatbot/' },
      { text: 'PLN', link: '/pln/' },
      { text: 'Regressão', link: '/regressao/' },
      { text: 'Blog', link: 'http://giseldo.github.io' },
    ],

    sidebar: {
      '/pln/': [
        {
          text: 'PLN',
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
          text: 'Regressao',
          items: [
            { text: 'Introcução', link: '/regressao/' },
            { text: 'Vídeo', link: '/regressao/2-video' },
            { text: 'Exercício', link: '/regressao/3-exercicios' },
          ]
        }
      ]
    },


    socialLinks: [
      { icon: 'github', link: 'https://github.com/giseldo' }
    ]
  }
})


