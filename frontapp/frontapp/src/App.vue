<template>
  <div>
    <input type="file" @change="onFileChange">
    <button @click="upload">Upload</button>
  </div>
</template>

<script>
import { uploadFile } from './components/s3';

export default {
  data() {
    return {
      selectedFile: null
    };
  },
  methods: {
    onFileChange(e) {
      this.selectedFile = e.target.files[0];
    },
    async upload() {
      try {
        const response = await uploadFile(this.selectedFile);
        console.log('File uploaded successfully', response);
      } catch (error) {
        console.error('There was an error uploading the file', error);
      }
    }
  }
};
</script>