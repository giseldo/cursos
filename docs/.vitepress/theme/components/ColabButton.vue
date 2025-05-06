<template>
  <a :href="href" target="_blank" rel="noopener noreferrer">
    <img src="/pln/fig/colab.png" alt="Open In Colab" />
  </a>
</template>

<script setup>
const props = defineProps({
  href: {
    type: String,
    required: true
  }
});
</script>
