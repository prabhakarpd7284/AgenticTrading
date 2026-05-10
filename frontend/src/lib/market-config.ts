/**
 * Market configuration constants — single source of truth for index metadata.
 *
 * Lot sizes:  NSE circular FAOP70616, effective Jan 2026.
 * Expiry days: SEBI standardisation Sep 2025 — NSE=Tue, BSE=Thu.
 * Tokens:     Angel One SmartAPI index tokens.
 *
 * Import from here instead of hardcoding in individual pages.
 */

/* ------------------------------------------------------------------ */
/* Index metadata                                                      */
/* ------------------------------------------------------------------ */

export interface IndexMeta {
  symbol: string;
  label: string;
  token: string;           // Angel One SmartAPI token
  exchange: "NSE" | "BSE";
  lotSize: number;
  expiryWeekday: number;   // Mon=0 … Sun=6
  /** True if only monthly expiry (no weekly contracts). */
  monthlyOnly: boolean;
}

export const INDICES: Record<string, IndexMeta> = {
  NIFTY: {
    symbol: "NIFTY",
    label: "NIFTY 50",
    token: "99926000",
    exchange: "NSE",
    lotSize: 65,
    expiryWeekday: 1,      // Tuesday
    monthlyOnly: false,
  },
  BANKNIFTY: {
    symbol: "BANKNIFTY",
    label: "BANK NIFTY",
    token: "99926009",
    exchange: "NSE",
    lotSize: 30,
    expiryWeekday: 1,      // Tuesday (monthly = last Tue)
    monthlyOnly: true,      // no weekly contracts
  },
  SENSEX: {
    symbol: "SENSEX",
    label: "SENSEX",
    token: "99919000",
    exchange: "BSE",
    lotSize: 20,
    expiryWeekday: 3,      // Thursday
    monthlyOnly: false,
  },
} as const;

/** Ordered list for UI toggles. */
export const INDEX_LIST = ["NIFTY", "BANKNIFTY", "SENSEX"] as const;

/* ------------------------------------------------------------------ */
/* Derived helpers                                                     */
/* ------------------------------------------------------------------ */

export function getLotSize(underlying: string): number {
  return INDICES[underlying]?.lotSize ?? 65;
}

export function getExpiryWeekday(underlying: string): number {
  return INDICES[underlying]?.expiryWeekday ?? 1;
}

export function isMonthlyOnly(underlying: string): boolean {
  return INDICES[underlying]?.monthlyOnly ?? false;
}

/* ------------------------------------------------------------------ */
/* Sector mapping (NIFTY50 / common NSE symbols)                       */
/* ------------------------------------------------------------------ */

export const SECTOR_MAP: Record<string, string> = {
  // Banking & Finance
  HDFCBANK: "Banking", ICICIBANK: "Banking", SBIN: "Banking",
  KOTAKBANK: "Banking", AXISBANK: "Banking", INDUSINDBK: "Banking",
  BAJFINANCE: "Finance", BAJAJFINSV: "Finance", HDFCLIFE: "Finance",
  SBILIFE: "Finance", MFSL: "Finance",
  // IT
  TCS: "IT", INFY: "IT", WIPRO: "IT", HCLTECH: "IT",
  TECHM: "IT", LTIM: "IT",
  // Auto
  MARUTI: "Auto", TATAMOTORS: "Auto", "M&M": "Auto",
  "BAJAJ-AUTO": "Auto", EICHERMOT: "Auto", HEROMOTOCO: "Auto",
  // Metals & Mining
  TATASTEEL: "Metals", HINDALCO: "Metals", JSWSTEEL: "Metals",
  COALINDIA: "Metals",
  // Pharma & Healthcare
  SUNPHARMA: "Pharma", DRREDDY: "Pharma", CIPLA: "Pharma",
  APOLLOHOSP: "Healthcare", DIVISLAB: "Pharma",
  // FMCG
  ITC: "FMCG", HINDUNILVR: "FMCG", NESTLEIND: "FMCG",
  TATACONSUM: "FMCG", BRITANNIA: "FMCG",
  // Energy / Power / Infra
  RELIANCE: "Energy", ONGC: "Energy", NTPC: "Power",
  POWERGRID: "Power", ADANIENT: "Conglomerate", ADANIPORTS: "Infra",
  // Others
  ULTRACEMCO: "Cement", GRASIM: "Cement", SHREECEM: "Cement",
  TITAN: "Consumer", ASIANPAINT: "Consumer",
  LT: "Infra", BHARTIARTL: "Telecom",
};

export function getSector(symbol: string): string {
  return SECTOR_MAP[symbol] ?? "Other";
}
