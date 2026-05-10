# AlphaDesk — Design System

## 1. Principles

AlphaDesk is a professional workstation. Traders stare at it for hours. Every pixel earns its place.

1. **Data is primary.** Numbers lead, chrome recedes. Tabular-lining figures, tight tracking, monospace for all quantities.
2. **Calm by default, loud by exception.** Neutral UI; colour only for P&L, risk, and live events.
3. **Fast feedback.** Loading states ≤ 150 ms skeleton, transitions ≤ 200 ms. Never block on async work.
4. **Dark-first.** Designed for low-light desks; a light mode is provided but is not primary.
5. **Accessibility is not optional.** WCAG 2.1 AA minimum on every screen. Keyboard-first.
6. **Composable, not rigid.** Primitives ship unstyled behaviour; composition happens at the feature layer.

## 2. Tokens

All tokens are defined as CSS custom properties on `:root` and overridden in `[data-theme="light"]`. Tailwind reads them through the `theme.extend` map in `tailwind.config.js`.

### 2.1 Colour

Neutral scale — the UI spine. Always used by semantic role, never by number directly in components.

| Role | Dark value | Light value | Used for |
|---|---|---|---|
| `bg` | `#0B0D10` | `#FAFAFA` | App background |
| `surface` | `#121519` | `#FFFFFF` | Card surface, panel |
| `surface-2` | `#171B21` | `#F4F4F5` | Raised element |
| `border` | `#23272E` | `#E5E7EB` | Dividers, hairlines |
| `border-strong` | `#2D333C` | `#D4D4D8` | Emphasised borders |
| `fg` | `#ECEFF3` | `#0B0D10` | Primary text |
| `fg-muted` | `#A0A6AE` | `#52525B` | Secondary text |
| `fg-subtle` | `#6B7079` | `#8A8F98` | Placeholders, captions |

Brand — the spine of interactive colour. Indigo-tinted cyan.

| Token | Value | Use |
|---|---|---|
| `brand-50` … `brand-950` | full 11-stop scale | Gradients, muted backgrounds |
| `accent` | `#6AE3D1` | Primary CTA, focus ring, brand marks |
| `accent-hover` | `#8BEEDC` | Hover state |
| `accent-pressed` | `#4DC8B5` | Active state |
| `accent-fg` | `#06201C` | Text on accent |

Semantic — always used by role, never hard-coded.

| Role | Token | Dark | Light |
|---|---|---|---|
| P&L positive | `pnl-up` | `#10B981` | `#059669` |
| P&L negative | `pnl-down` | `#F43F5E` | `#E11D48` |
| Warning | `warn` | `#F59E0B` | `#D97706` |
| Info | `info` | `#38BDF8` | `#0284C7` |
| Danger | `danger` | `#F43F5E` | `#DC2626` |

### 2.2 Type

Three families:

| Family | Use | Fallback |
|---|---|---|
| `Inter var` | UI text | system-ui, sans-serif |
| `JetBrains Mono` | All numerics, codes, tokens | ui-monospace |
| `Space Grotesk` | Display / hero | Inter var |

Scale (all in rem, assume 16 px root):

| Name | Size | Line-height | Tracking | Use |
|---|---|---|---|---|
| `display-xl` | 3.5 rem / 56 | 1.05 | -0.025em | Marketing hero |
| `display-lg` | 2.5 rem / 40 | 1.1  | -0.025em | Auth / onboarding |
| `h1` | 1.875 rem / 30 | 1.2 | -0.015em | Page title |
| `h2` | 1.5 rem / 24 | 1.25 | -0.01em | Section |
| `h3` | 1.25 rem / 20 | 1.3 | -0.005em | Subsection |
| `body-lg` | 1 rem / 16 | 1.55 | 0 | Primary body |
| `body` | 0.875 rem / 14 | 1.5 | 0 | Default UI |
| `body-sm` | 0.8125 rem / 13 | 1.45 | 0 | Dense tables |
| `caption` | 0.75 rem / 12 | 1.4 | 0.01em | Metadata, captions |
| `num-lg` | 1.75 rem / 28 | 1.1 | -0.01em, tabular | KPI number |
| `num` | 1 rem / 16 | 1.2 | 0, tabular | Inline number |
| `num-sm` | 0.8125 rem / 13 | 1.2 | 0, tabular | Table cell |

All numeric styles use `font-feature-settings: "tnum" 1, "zero" 1`.

### 2.3 Spacing

4-based scale. Components align to a 4-px grid; dense tables to 2-px.

| Token | Value |
|---|---|
| `space-0` | 0 |
| `space-0.5` | 2 px |
| `space-1` | 4 px |
| `space-1.5` | 6 px |
| `space-2` | 8 px |
| `space-3` | 12 px |
| `space-4` | 16 px |
| `space-5` | 20 px |
| `space-6` | 24 px |
| `space-8` | 32 px |
| `space-10` | 40 px |
| `space-12` | 48 px |
| `space-16` | 64 px |
| `space-24` | 96 px |

### 2.4 Radius

| Token | Value | Use |
|---|---|---|
| `radius-xs` | 4 px | Inputs, chips |
| `radius-sm` | 6 px | Buttons |
| `radius` | 8 px | Cards, dialogs |
| `radius-lg` | 12 px | Modal |
| `radius-full` | 9999 px | Avatar, pill |

### 2.5 Elevation

Shadows are sparingly used; dark surfaces rely on border + bg contrast.

| Token | Value |
|---|---|
| `shadow-sm` | `0 1px 2px rgba(0,0,0,0.25)` |
| `shadow` | `0 2px 6px rgba(0,0,0,0.32), 0 1px 2px rgba(0,0,0,0.18)` |
| `shadow-lg` | `0 12px 32px rgba(0,0,0,0.45)` |
| `shadow-glow` | `0 0 0 1px var(--accent), 0 0 0 4px rgba(106,227,209,0.16)` (focus) |

### 2.6 Motion

Durations scaled for density; prefer instant (`0ms`) for selections, short (`120ms`) for hover.

| Token | Duration | Easing |
|---|---|---|
| `motion-0` | 0 ms | — |
| `motion-1` | 120 ms | `cubic-bezier(0.2, 0, 0, 1)` |
| `motion-2` | 180 ms | `cubic-bezier(0.2, 0, 0, 1)` |
| `motion-3` | 240 ms | `cubic-bezier(0.2, 0, 0, 1)` |
| `motion-page` | 320 ms | `cubic-bezier(0.16, 1, 0.3, 1)` |

Respect `prefers-reduced-motion` — all durations collapse to 0 and transforms are disabled.

### 2.7 Z-index scale

| Token | Value | Use |
|---|---|---|
| `z-base` | 0 | Page |
| `z-sticky` | 20 | Sticky table header, AppShell top bar |
| `z-overlay` | 40 | Dropdown, popover |
| `z-dialog` | 50 | Modal, drawer |
| `z-toast` | 60 | Toast |
| `z-tooltip` | 70 | Tooltip |

## 3. Components

All primitives live at `frontend/src/components/ui/`. Each exposes a `data-*` state so Tailwind can style via `data-[state=open]:`.

### 3.1 Button

**Variants:** `primary`, `secondary`, `ghost`, `destructive`, `link`.
**Sizes:** `sm` (28 px), `md` (36 px, default), `lg` (44 px).
**States:** default · hover · active · disabled · loading.
**A11y:** native `<button>`; `aria-busy` while loading; focus ring = 2px accent + 2px bg halo.

### 3.2 Input / Textarea / Select

32-px (`sm`), 36-px (`md`). Label is required; placeholder is hint, never the label. Supports `leading`, `trailing`, `invalid`, `hint`, `error` props. Errors announced via `aria-describedby`.

### 3.3 Card / Panel

`Card` — `radius`, 1-px border, `surface` bg. Composes `CardHeader`, `CardTitle`, `CardDescription`, `CardContent`, `CardFooter`. No drop shadow in dark theme.

### 3.4 Badge / Tag / Chip

- `Badge` — status: `neutral`, `brand`, `success`, `warning`, `danger`, `info`. 18 px tall.
- `Chip` — removable tag for filters.

### 3.5 KPI

Specialised card for headline metrics. Composes label, number (num-lg), delta (`pnl-up` / `pnl-down` badge), optional sparkline. Numbers animate in on mount with `motion-3`.

### 3.6 Table / DataGrid

- Sticky header, zebra optional, 40-px row (normal) or 32-px (dense).
- Column types: `text`, `num`, `money`, `delta`, `timestamp`, `status`, `action`.
- Sort ⇄, filter ▽, group chips at top.
- Row selection with `Space`; arrow-keys navigate.
- Virtualised for ≥ 200 rows.

### 3.7 Dialog + Drawer

Radix-based. Backdrop = `rgba(0,0,0,0.6)`. Closes on `Esc`; focus trapped; returns focus to trigger. `DialogTitle` + `DialogDescription` are required for screen readers.

### 3.8 Toast

`sonner`-based. Positions: `top-right` (non-critical) and `bottom-center` (critical). Contains: icon, title, optional description, optional action, dismiss. `role="status"` for info, `role="alert"` for error.

### 3.9 Tabs

Radix-based. Horizontal and vertical. 36-px trigger; underline accent on active.

### 3.10 Tooltip

Radix-based. Delay = 250 ms. Arrow; `surface-2` bg; `caption` type.

### 3.11 Command palette (⌘K)

Global search + action launcher. Groups: `Navigate`, `Agents`, `Positions`, `Journal`, `Shortcuts`. Debounced query; keyboard-only; uses `cmdk`.

### 3.12 Skeleton / EmptyState / ErrorState

- `Skeleton` — shimmering block with `motion-3` loop; respects reduced-motion (solid fill fallback).
- `EmptyState` — icon + title + description + primary CTA.
- `ErrorState` — icon + title + description + retry CTA + "contact support" link.

## 4. Patterns

### 4.1 Form

Stacked labels; inline validation on blur; error message below field; submit button right-aligned with `Cancel` secondary left of it; `Enter` submits; `Esc` cancels.

### 4.2 P&L display

Always signed (`+1,245.25` / `−820.10`), always monospace, always coloured `pnl-up` / `pnl-down`. Zero values are `fg-muted`. Never truncate without a tooltip showing the full value.

### 4.3 Live data

Three indicators:
1. Connection badge in top bar: `Live`, `Reconnecting…`, `Offline`.
2. Updated-cell pulse (120 ms `shadow-glow` on change) — disabled under reduced-motion.
3. Stale-data warning at 10 s since last tick; critical at 30 s.

### 4.4 Risk-breach affordance

When a risk check fails:
- Red banner at top of page `role="alert"`, dismissible only by resolving.
- The `Place order` CTA switches to `Blocked by RiskGuard` (destructive, disabled).
- The blocked reason is always quoted verbatim under the CTA.

### 4.5 Navigation

Left sidebar (collapsible to icon rail). Sections: `Overview`, `Agents`, `Positions`, `Journal`, `Strategies`, `Backtests`, `Broker`, `Settings`. `⌘K` opens command palette; `g` then letter jumps to section.

## 5. Accessibility baseline (WCAG 2.1 AA)

| Area | Requirement |
|---|---|
| Contrast | Text ≥ 4.5:1; large text ≥ 3:1; non-text UI ≥ 3:1. All tokens above are audited. |
| Focus | Visible, 2 px accent ring with 2 px bg halo on every interactive element. |
| Keyboard | Every action reachable without a mouse. No keyboard traps. |
| Motion | Honour `prefers-reduced-motion`. No parallax. No autoplay > 5 s. |
| Targets | ≥ 24 × 24 CSS px; ≥ 44 × 44 for touch on mobile. |
| Forms | Each field has a `<label for>`. Errors announced. |
| Live regions | Use `aria-live` for P&L ticks (polite) and risk alerts (assertive). |
| Dialogs | `role="dialog"`, focus trap, `aria-labelledby`, `aria-describedby`. |
| Tables | `<caption>`, `scope="col"`, sortable cols use `aria-sort`. |

## 6. UX-copy voice

- Direct, never chirpy. We speak to experienced operators.
- Show, don't explain. "2 open positions" beats "You currently have 2 open positions".
- Precise verbs for actions: `Place order`, `Cancel order`, `Close position`, `Run backtest`.
- Error messages always have: what happened, why, what to do. No "Something went wrong."
- Numbers come with unit: `₹ 5,00,000`, `+1.24 %`, `14 s`. Format Indian numerals in the Indian number system (lakh/crore) by default, with Western grouping available as a preference.

## 7. Do's and don'ts

| ✅ Do | ❌ Don't |
|---|---|
| Use semantic tokens (`bg-surface`) | Use raw palette (`bg-zinc-900`) |
| Reserve green/red for P&L | Use green/red as UI decoration |
| Mono for every quantity | Proportional digits in tables |
| Announce live P&L in an `aria-live` region | Only colour-code changes |
| Provide keyboard shortcut for every primary action | Rely solely on a context menu |

## 8. Component status

Current coverage (first cut; see `frontend/src/components/ui/`):

| Component | Variants | States | Docs | Score |
|---|---|---|---|---|
| Button | ✅ | ✅ | ✅ | 10/10 |
| Input | ✅ | ✅ | ✅ | 10/10 |
| Card | ✅ | ✅ | ✅ | 10/10 |
| Badge | ✅ | ✅ | ✅ | 10/10 |
| KPI | ✅ | ✅ | ✅ | 10/10 |
| Table | ✅ | ✅ | ✅ | 9/10 (virtualised list follow-up) |
| Dialog | ✅ | ✅ | ⚠️ | 8/10 |
| Toast | via sonner | ✅ | ✅ | 9/10 |
| Tabs | ✅ | ✅ | ⚠️ | 8/10 |
| Tooltip | ✅ | ✅ | ⚠️ | 8/10 |
| CommandPalette | ✅ | ✅ | ✅ | 9/10 |
| Skeleton | ✅ | ✅ | ✅ | 10/10 |
| EmptyState | ✅ | ✅ | ✅ | 10/10 |
| ErrorState | ✅ | ✅ | ✅ | 10/10 |
