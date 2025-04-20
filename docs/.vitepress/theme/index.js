import DefaultTheme from 'vitepress/theme';

import Tts from './components/Tts.vue'

import './custom.css'

export default {
  ...DefaultTheme,
  
  enhanceApp({ app, router }) {
    app.component('Tts', Tts);

    if (typeof window !== 'undefined') {
      // Adiciona o evento para capturar teclas pressionadas
      window.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowRight') {
          // Navegar para a próxima página
          const nextLink = document.querySelector('.next');
          if (nextLink) {
            nextLink.click();
          }
        } else if (event.key === 'ArrowLeft') {
          // Navegar para a página anterior
          const prevLink = document.querySelector('.prev');
          if (prevLink) {
            prevLink.click();
          }
        }
      });
    }
  },
};
