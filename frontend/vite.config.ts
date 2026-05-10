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
    port: 5173,
    proxy: {
      // Legacy sqlite bridge runs in its own Django process on :8001.
      // The catch-all /api rule picks up everything else and points at
      // the v2 Postgres stack on :8000. Order matters — /api/v1/legacy/
      // must match before the generic /api rule.
      "/api/v1/legacy": { target: "http://localhost:8001", changeOrigin: true },
      "/api":           { target: "http://localhost:8000", changeOrigin: true },
      "/ws":            { target: "ws://localhost:8000",   ws: true },
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
