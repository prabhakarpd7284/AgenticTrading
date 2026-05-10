/**
 * Vitest global setup.
 *
 * Runs once per test file before any test. Purpose:
 *  - Register jest-dom matchers (`toBeInTheDocument`, `toHaveAttribute`, …).
 *  - Shim browser APIs that jsdom doesn't implement but our design system uses
 *    (matchMedia for theme + reduced-motion, IntersectionObserver for the
 *    virtualised DataTable, ResizeObserver for charts).
 *  - Reset timers after each test so animation-driven UI can't bleed state.
 */
import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// --- matchMedia -----------------------------------------------------------
// Our theme toggle and prefers-reduced-motion guards use matchMedia.
if (!window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(() => false),
    }),
  });
}

// --- IntersectionObserver ------------------------------------------------
class IOStub {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
  takeRecords = vi.fn(() => []);
  root = null;
  rootMargin = "";
  thresholds: ReadonlyArray<number> = [];
}
if (!("IntersectionObserver" in window)) {
  // @ts-expect-error -- shim
  window.IntersectionObserver = IOStub;
}

// --- ResizeObserver ------------------------------------------------------
class ROStub {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}
if (!("ResizeObserver" in window)) {
  // @ts-expect-error -- shim
  window.ResizeObserver = ROStub;
}

// --- scrollTo / scrollIntoView ------------------------------------------
// Radix primitives call these during focus management.
if (!("scrollTo" in window)) {
  // @ts-expect-error -- shim
  window.scrollTo = vi.fn();
}
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = vi.fn();
}

// --- clean up React Testing Library after every test --------------------
afterEach(() => {
  cleanup();
});
