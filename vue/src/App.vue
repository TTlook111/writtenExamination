<template>
  <UploadResume />
</template>

<script>
import UploadResume from './components/UploadResume.vue'

export default {
  name: 'App',
  components: {
    UploadResume
  }
}
</script>

<style>
body {
  margin: 0;
  padding: 0;
  background-color: #f0f2f5;
}
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
</style>
