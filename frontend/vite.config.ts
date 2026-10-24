import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The UI talks to the local FastAPI process. In dev, Vite proxies /api to it;
// in "demo mode" the backend serves the built UI itself on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
