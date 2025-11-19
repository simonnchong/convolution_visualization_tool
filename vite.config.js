import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // This dot-slash tells it to look for files "right here"
  // It prevents 404 errors caused by wrong repository names
  base: './', 
})
