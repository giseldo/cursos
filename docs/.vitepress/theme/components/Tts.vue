<!-- c:\Projetos\cursos\docs\chatbotbook\components\Tts.vue -->
<template>
  <div class="tts-controls">
    <button @click="speakText" :disabled="isSpeaking || !isSupported" :title="buttonTitle">
      <!-- Ícone de alto-falante (SVG simples) -->
      <svg v-if="!isSpeaking" xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-volume-up-fill" viewBox="0 0 16 16">
        <path d="M11.536 14.01A8.47 8.47 0 0 0 14.026 8a8.47 8.47 0 0 0-2.49-6.01l-.708.707A7.476 7.476 0 0 1 13.025 8c0 2.071-.84 3.946-2.197 5.303z"/>
        <path d="M10.121 12.596A6.48 6.48 0 0 0 12.025 8a6.48 6.48 0 0 0-1.904-4.596l-.707.707A5.483 5.483 0 0 1 11.025 8a5.483 5.483 0 0 1-1.61 3.89z"/>
        <path d="M8.707 11.182A4.486 4.486 0 0 0 10.025 8a4.486 4.486 0 0 0-1.318-3.182L8 5.525A3.489 3.489 0 0 1 9.025 8 3.49 3.49 0 0 1 8 10.475zM6.717 3.55A.5.5 0 0 1 7 4v8a.5.5 0 0 1-.812.39L3.825 10.5H1.5A.5.5 0 0 1 1 10V6a.5.5 0 0 1 .5-.5h2.325l2.363-1.89a.5.5 0 0 1 .529-.06"/>
      </svg>
      <!-- Ícone de carregamento/falando (SVG simples) -->
      <svg v-else xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-soundwave" viewBox="0 0 16 16">
        <path fill-rule="evenodd" d="M8.5 2a.5.5 0 0 1 .5.5v11a.5.5 0 0 1-1 0v-11a.5.5 0 0 1 .5-.5m-2 2a.5.5 0 0 1 .5.5v7a.5.5 0 0 1-1 0v-7a.5.5 0 0 1 .5-.5m4 0a.5.5 0 0 1 .5.5v7a.5.5 0 0 1-1 0v-7a.5.5 0 0 1 .5-.5m-6 1.5a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 1 .5-.5m8 0a.5.5 0 0 1 .5.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 1 .5-.5m-10 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-1 0v-1a.5.5 0 0 1 .5-.5m12 0a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-1 0v-1a.5.5 0 0 1 .5-.5"/>
      </svg>
      <span class="button-text">{{ isSpeaking ? 'Parar' : 'Ouvir' }}</span>
    </button>
    <p v-if="!isSupported" class="support-warning">Seu navegador não suporta a API de Fala.</p>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, computed } from 'vue';

// Define a prop 'text' que o componente receberá
const props = defineProps({
  text: {
    type: String,
    required: true,
    default: '' // Valor padrão para evitar erros se não for passado
  }
});

// Referência reativa para controlar o estado de fala
const isSpeaking = ref(false);
// Referência para verificar se a API é suportada
const isSupported = ref(false);
// Armazena a instância da utterance para poder cancelá-la
let currentUtterance = null;

// Acessa a API de síntese de fala do navegador
const synth = window.speechSynthesis;

// Função para iniciar ou parar a fala
const speakText = () => {
  if (!isSupported.value) return; // Não faz nada se não for suportado

  if (isSpeaking.value && synth && currentUtterance) {
    // Se já está falando, cancela a fala atual
    synth.cancel();
    // O evento 'end' ou 'error' deve resetar isSpeaking, mas garantimos aqui
    isSpeaking.value = false;
    currentUtterance = null;
  } else if (props.text && synth) {
    // Se não está falando e há texto, inicia a fala
    // Cancela qualquer fala anterior que possa ter ficado presa
    synth.cancel();

    // Cria uma nova instância de SpeechSynthesisUtterance
    const utterance = new SpeechSynthesisUtterance(props.text);
    currentUtterance = utterance; // Armazena a referência

    // Define o idioma (ajuste conforme necessário)
    utterance.lang = 'pt-BR';

    // Define callbacks para os eventos da fala
    utterance.onstart = () => {
      isSpeaking.value = true;
    };

    utterance.onend = () => {
      isSpeaking.value = false;
      currentUtterance = null; // Limpa a referência
    };

    utterance.onerror = (event) => {
      console.error("Erro na síntese de fala:", event.error);
      isSpeaking.value = false;
      currentUtterance = null; // Limpa a referência
    };

    // Inicia a fala
    synth.speak(utterance);
  }
};

// Título dinâmico para o botão
const buttonTitle = computed(() => {
    if (!isSupported.value) {
        return "Síntese de fala não suportada";
    }
    return isSpeaking.value ? "Parar leitura" : "Ouvir o texto";
});

// Verifica o suporte da API quando o componente é montado
onMounted(() => {
  isSupported.value = 'speechSynthesis' in window && typeof SpeechSynthesisUtterance !== 'undefined';
  // Garante que qualquer fala pendente seja cancelada ao montar
  if (synth) {
      synth.cancel();
  }
});

// Garante que a fala seja cancelada se o componente for desmontado
onBeforeUnmount(() => {
  if (isSpeaking.value && synth) {
    synth.cancel();
  }
});
</script>

<style scoped>
.tts-controls {
  display: inline-flex; /* Para alinhar botão e texto de aviso lado a lado se necessário */
  align-items: center;
  gap: 8px; /* Espaço entre o botão e o aviso */
  margin: 5px 0; /* Pequena margem vertical */
}

button {
  display: inline-flex; /* Alinha ícone e texto dentro do botão */
  align-items: center;
  gap: 5px; /* Espaço entre ícone e texto */
  padding: 4px 8px;
  font-size: 0.9em;
  cursor: pointer;
  border: 1px solid #ccc;
  border-radius: 4px;
  background-color: #f0f0f0;
  transition: background-color 0.2s ease;
}

button:hover:not(:disabled) {
  background-color: #e0e0e0;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.support-warning {
  color: #dc3545; /* Cor de aviso (vermelho) */
  font-size: 0.8em;
  margin: 0;
}

/* Estilo opcional para o texto dentro do botão, se necessário */
/* .button-text { } */
</style>
