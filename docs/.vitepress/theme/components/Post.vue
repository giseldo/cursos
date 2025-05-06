<script setup>
// Here we're using the composition API with the <script setup> feature.
// Everything in this script runs in the Vue component's setup function and
// any declared variables can be accessed from the component's template.

// defineProps lets you access the component's props from the setup function.
import { defineProps } from 'vue'

// Set the props as variables so we can access them in the template.
const { title, link } = defineProps(['title', 'link'])
</script>

<template>
<div class="post">
  <a :href="link"><p class="title">{{ title }}</p></a>
  <p class="description"><slot></slot></p>
</div>
</template>

<style scoped>
.post {
  border: 3px solid #333;
  margin: 1rem;
  padding: 1rem;
  border-radius: 3rem;
}
.title {
  font-size: 3rem;
  font-weight: 300;
}
.description {
  font-size: 1.5rem;
  font-weight: 200;
}
</style>