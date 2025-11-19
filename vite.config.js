import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // IMPORTANT: This must match your repository name exactly, with slashes
  base: '/convolution_visualization_tool/', 
})
