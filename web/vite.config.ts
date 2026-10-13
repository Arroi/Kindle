import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { readFileSync } from "node:fs";
import { defineConfig } from "vite";

const { version } = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8"));

const backend = process.env.KAGGLEUI_DEV_BACKEND ?? "http://127.0.0.1:8890";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: { __APP_VERSION__: JSON.stringify(version) },
  build: {
    outDir: "../kaggleui/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
  },
  server: {
    port: 5173,
    proxy: {
      "/_kui": { target: backend, ws: true, changeOrigin: false },
      "/v1": { target: backend, changeOrigin: false },
      "/api": { target: backend, changeOrigin: false },
      "/gradio": { target: backend, ws: true, changeOrigin: false },
    },
  },
});
