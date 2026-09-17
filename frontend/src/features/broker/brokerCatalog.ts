/**
 * Broker catalog — UI metadata for each broker plugin the backend supports.
 *
 * The list of registered brokers is dynamic (GET /api/v1/brokers/available/),
 * but each broker needs a credential form that asks for the right fields and a
 * blurb explaining its setup flow. Keeping that as data (not branched UI) keeps
 * the page rendering simple.
 *
 * If the backend reports a broker not in this catalog (future plugin), the UI
 * falls back to a generic free-form JSON credentials field so the operator can
 * still link it.
 */

export type FieldKind = "text" | "password" | "url" | "secret";

export interface BrokerField {
  key: string;
  label: string;
  kind: FieldKind;
  placeholder?: string;
  hint?: string;
  required?: boolean;
}

export interface MetaField {
  key: string;
  label: string;
  placeholder?: string;
  hint?: string;
}

export interface BrokerSpec {
  name: string;            // backend identifier — matches /brokers/available/
  label: string;           // display name
  blurb: string;           // one-liner shown on the card
  setupUrl?: string;       // link to the broker's API console for getting creds
  recommended?: boolean;
  fields: BrokerField[];          // direct credential fields (paste-the-token path)
  metaFields?: MetaField[];
  dailyTokenRefresh?: boolean;    // true for Kite/Fyers — show the warning
  oauth?: {
    // OAuth-redirect flow — UI sends `handshake` to /oauth/start/, broker
    // login page handles the rest, callback persists the access_token.
    handshakeFields: BrokerField[];
    redirectHelp: string;         // exact redirect URL the operator pastes into the broker app portal
  };
}

export const BROKER_CATALOG: Record<string, BrokerSpec> = {
  angel_one: {
    name: "angel_one",
    label: "Angel One SmartAPI",
    blurb: "Certified integration. NSE/NFO/BSE/BFO/MCX cash + options.",
    setupUrl: "https://smartapi.angelbroking.com/",
    recommended: true,
    fields: [
      { key: "api_key", label: "API key", kind: "text", required: true,
        placeholder: "e.g. ALvA15GL",
        hint: "From SmartAPI dashboard → App details." },
      { key: "client_code", label: "Client code", kind: "text", required: true,
        placeholder: "e.g. A100123",
        hint: "Your Angel One trading client id (uppercase + digits)." },
      { key: "password", label: "MPIN / Password", kind: "password", required: true,
        hint: "Angel One MPIN — the 4-digit transaction PIN, not the login password." },
      { key: "totp_secret", label: "TOTP secret", kind: "secret", required: true,
        placeholder: "base32 string (20–32 chars)",
        hint: "Setup TOTP under Profile → Security; copy the base32 secret, not the 6-digit code." },
    ],
    metaFields: [
      { key: "account_alias", label: "Account alias", placeholder: "e.g. Personal" },
    ],
  },

  zerodha: {
    name: "zerodha",
    label: "Zerodha Kite Connect",
    blurb: "Read-only positions/holdings/margin. Daily access-token re-login via Kite login.",
    setupUrl: "https://developers.kite.trade/apps",
    dailyTokenRefresh: true,
    fields: [],   // OAuth-only — no paste-the-token form
    oauth: {
      handshakeFields: [
        { key: "api_key", label: "API key", kind: "text", required: true,
          hint: "From Kite Connect → My apps." },
        { key: "api_secret", label: "API secret", kind: "secret", required: true,
          hint: "Used to sign the request_token exchange." },
      ],
      redirectHelp:
        "Register this redirect URL in your Kite app: " +
        "http://127.0.0.1:5173/api/v1/brokers/zerodha/oauth/callback/",
    },
    metaFields: [
      { key: "account_alias", label: "Account alias", placeholder: "e.g. Personal" },
    ],
  },

  fyers: {
    name: "fyers",
    label: "Fyers API v3",
    blurb: "Read-only positions/holdings/funds. Daily access-token re-login via Fyers login.",
    setupUrl: "https://myapi.fyers.in/",
    dailyTokenRefresh: true,
    fields: [],   // OAuth-only
    oauth: {
      handshakeFields: [
        { key: "app_id", label: "App id", kind: "text", required: true,
          placeholder: "e.g. XYZA-100" },
        { key: "secret_key", label: "Secret key", kind: "secret", required: true },
      ],
      redirectHelp:
        "Register this redirect URL in your Fyers app: " +
        "http://127.0.0.1:5173/api/v1/brokers/fyers/oauth/callback/",
    },
    metaFields: [
      { key: "account_alias", label: "Account alias", placeholder: "e.g. Personal" },
    ],
  },

  paper: {
    name: "paper",
    label: "Paper broker",
    blurb: "Virtual fills against live market prices. No credentials required.",
    fields: [
      // Single sentinel so the connect endpoint doesn't reject the empty body.
      { key: "mode", label: "Mode", kind: "text", placeholder: "paper",
        hint: "Leave as `paper`." },
    ],
    metaFields: [
      { key: "account_alias", label: "Account alias", placeholder: "e.g. Default paper" },
    ],
  },
};

export function getBrokerSpec(name: string): BrokerSpec {
  if (name in BROKER_CATALOG) return BROKER_CATALOG[name];
  return {
    name,
    label: name,
    blurb: `Generic broker plugin (${name}) — no UI catalog entry yet.`,
    fields: [
      { key: "credentials_json", label: "Credentials JSON", kind: "secret", required: true,
        hint: "Raw JSON object the adapter expects (will be sent as the credentials body)." },
    ],
  };
}
