/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    container: { center: true, padding: "1rem", screens: { "2xl": "1400px" } },
    extend: {
      // ---------- colour tokens (CSS variables set in globals.css) ----------
      colors: {
        bg:            "rgb(var(--bg) / <alpha-value>)",
        surface:       "rgb(var(--surface) / <alpha-value>)",
        "surface-2":   "rgb(var(--surface-2) / <alpha-value>)",
        "surface-3":   "rgb(var(--surface-3) / <alpha-value>)",
        border:        "rgb(var(--border) / <alpha-value>)",
        "border-strong": "rgb(var(--border-strong) / <alpha-value>)",
        fg:            "rgb(var(--fg) / <alpha-value>)",
        "fg-muted":    "rgb(var(--fg-muted) / <alpha-value>)",
        "fg-subtle":   "rgb(var(--fg-subtle) / <alpha-value>)",

        accent:        "rgb(var(--accent) / <alpha-value>)",
        "accent-hover":"rgb(var(--accent-hover) / <alpha-value>)",
        "accent-pressed":"rgb(var(--accent-pressed) / <alpha-value>)",
        "accent-fg":   "rgb(var(--accent-fg) / <alpha-value>)",

        "pnl-up":      "rgb(var(--pnl-up) / <alpha-value>)",
        "pnl-down":    "rgb(var(--pnl-down) / <alpha-value>)",
        warn:          "rgb(var(--warn) / <alpha-value>)",
        info:          "rgb(var(--info) / <alpha-value>)",
        danger:        "rgb(var(--danger) / <alpha-value>)",
      },

      // ---------- type ----------
      fontFamily: {
        sans: ['"Inter var"', "Inter", "system-ui", "ui-sans-serif", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
        display: ['"Space Grotesk"', '"Inter var"', "Inter", "sans-serif"],
      },
      fontSize: {
        "display-xl": ["3.5rem",   { lineHeight: "1.05", letterSpacing: "-0.025em", fontWeight: "600" }],
        "display-lg": ["2.5rem",   { lineHeight: "1.1",  letterSpacing: "-0.025em", fontWeight: "600" }],
        h1:           ["1.875rem", { lineHeight: "1.2",  letterSpacing: "-0.015em", fontWeight: "600" }],
        h2:           ["1.5rem",   { lineHeight: "1.25", letterSpacing: "-0.01em",  fontWeight: "600" }],
        h3:           ["1.25rem",  { lineHeight: "1.3",  letterSpacing: "-0.005em", fontWeight: "600" }],
        "body-lg":    ["1rem",     { lineHeight: "1.55", letterSpacing: "0" }],
        body:         ["0.875rem", { lineHeight: "1.5",  letterSpacing: "0" }],
        "body-sm":    ["0.8125rem",{ lineHeight: "1.45", letterSpacing: "0" }],
        caption:      ["0.75rem",  { lineHeight: "1.4",  letterSpacing: "0.01em" }],
        "num-lg":     ["1.75rem",  { lineHeight: "1.1",  letterSpacing: "-0.01em", fontWeight: "600" }],
        num:          ["1rem",     { lineHeight: "1.2",  letterSpacing: "0",       fontWeight: "500" }],
        "num-sm":     ["0.8125rem",{ lineHeight: "1.2",  letterSpacing: "0",       fontWeight: "500" }],
      },

      // ---------- spacing (4-based) ----------
      spacing: {
        0.5: "0.125rem", 1.5: "0.375rem", 2.5: "0.625rem", 3.5: "0.875rem",
        // Tailwind already ships 0..96; we add 18 + 22 for sidebar widths
        18: "4.5rem", 22: "5.5rem",
      },

      // ---------- radius ----------
      borderRadius: {
        xs: "4px",
        sm: "6px",
        DEFAULT: "8px",
        md: "8px",
        lg: "12px",
        xl: "16px",
      },

      // ---------- elevation ----------
      boxShadow: {
        sm:   "0 1px 2px rgba(0,0,0,0.25)",
        DEFAULT: "0 2px 6px rgba(0,0,0,0.32), 0 1px 2px rgba(0,0,0,0.18)",
        lg:   "0 12px 32px rgba(0,0,0,0.45)",
        glow: "0 0 0 1px rgb(var(--accent)), 0 0 0 4px rgb(var(--accent) / 0.16)",
      },

      // ---------- motion ----------
      transitionDuration: {
        0: "0ms", 120: "120ms", 180: "180ms", 240: "240ms", 320: "320ms",
      },
      transitionTimingFunction: {
        "out-expo": "cubic-bezier(0.16, 1, 0.3, 1)",
        "out-swift": "cubic-bezier(0.2, 0, 0, 1)",
      },
      keyframes: {
        "fade-in":  { "0%": { opacity: 0 }, "100%": { opacity: 1 } },
        "slide-up": { "0%": { opacity: 0, transform: "translateY(4px)" },
                      "100%": { opacity: 1, transform: "translateY(0)" } },
        shimmer:    { "0%": { backgroundPosition: "-400px 0" },
                      "100%": { backgroundPosition: "400px 0" } },
        pulseGlow:  { "0%,100%": { boxShadow: "0 0 0 0 rgb(var(--accent) / 0.0)" },
                      "50%":     { boxShadow: "0 0 0 3px rgb(var(--accent) / 0.28)" } },
      },
      animation: {
        "fade-in":  "fade-in 180ms cubic-bezier(0.16, 1, 0.3, 1)",
        "slide-up": "slide-up 240ms cubic-bezier(0.16, 1, 0.3, 1)",
        shimmer:    "shimmer 1.4s linear infinite",
        "pulse-glow": "pulseGlow 600ms cubic-bezier(0.2, 0, 0, 1) 1",
      },

      // ---------- z ----------
      zIndex: { base: "0", sticky: "20", overlay: "40", dialog: "50", toast: "60", tooltip: "70" },
    },
  },
  plugins: [
    // small accessibility helper: .sr-only already built in; add .focus-ring utility
    function ({ addUtilities, theme }) {
      addUtilities({
        ".focus-ring": {
          outline: "none",
          boxShadow: `0 0 0 2px rgb(var(--bg)), 0 0 0 4px rgb(var(--accent))`,
        },
        ".tabular": { fontVariantNumeric: "tabular-nums slashed-zero" },
      });
    },
  ],
};
