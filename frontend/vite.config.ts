/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  server: {
    // Bind both IPv4 and IPv6 so http://127.0.0.1:5173 works (browser OAuth
    // redirects from broker portals use the literal 127.0.0.1 — IPv6-only
    // binding returns ERR_CONNECTION_REFUSED for them).
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      // Phase 6 folded the legacy bridge into the v2 process on :8000 —
      // the old separate :8001 Django process and SQLite DB are gone.
      // Every /api/* (including /api/v1/legacy/*) now hits :8000.
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/ws":  { target: "ws://localhost:8000",   ws: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: true,
    coverage: {
      provider: "v8",
      reporter: ["text", "html", "lcov"],
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/**/*.d.ts", "src/main.tsx", "src/test/**"],
      thresholds: {
        // Kept intentionally low on the first cut; ratchet up per-folder as
        // coverage grows. Lines/branches targets per TESTING_STRATEGY.md.
        statements: 40,
        branches:   35,
        functions:  40,
        lines:      40,
      },
    },
  },
});
