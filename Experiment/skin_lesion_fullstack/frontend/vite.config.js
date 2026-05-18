import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    host: "0.0.0.0",
    port: Number(process.env.VITE_DEV_PORT || 6006),
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY || "http://127.0.0.1:5000",
        changeOrigin: true,
      },
    },
  },
});
