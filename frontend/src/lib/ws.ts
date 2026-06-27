/**
 * Thin, auto-reconnecting WebSocket client.
 *
 * Auth model: the Django ASGI app expects the JWT access token to arrive
 * *during the handshake* via a `Sec-WebSocket-Protocol` subprotocol — the
 * only way a browser can smuggle a bearer token into a WS upgrade request
 * (the spec forbids custom headers on WebSocket, and cookies don't work
 * when the SPA and API live on different origins).
 *
 * We offer two protocols: ["jwt", "<token>"]. The server's JWT middleware
 * picks the second string, validates it, and echoes back "jwt" so the
 * handshake completes.  If the token is missing/expired the server will
 * close the socket with code 4401 or 4403.
 *
 * Channels:
 *   /ws/ticks/   – subscribe/unsubscribe to tokens
 *   /ws/pnl/     – tenant-scoped P&L stream
 *   /ws/agents/<run_id>/ – per-run event stream
 *   /ws/alerts/  – notifications
 */
import { useAuthStore } from "@/stores/auth";

export type WSMessage = Record<string, unknown>;
export type WSHandler = (msg: WSMessage) => void;

interface Options {
  onOpen?: () => void;
  onClose?: (ev: CloseEvent) => void;
  onError?: (ev: Event) => void;
}

function resolveBase(): string {
  // In prod we set VITE_WS_URL explicitly (e.g. wss://api.alphadesk.io).
  // In dev we route through the Vite proxy, which forwards /ws/* to Django.
  const explicit = import.meta.env.VITE_WS_URL;
  if (explicit) return explicit.replace(/\/$/, "");
  return window.location.origin.replace(/^http/, "ws");
}

export function connect(path: string, onMessage: WSHandler, opts: Options = {}) {
  const base = resolveBase();
  const url = `${base}${path}`;

  let ws: WebSocket | null = null;
  let retries = 0;
  let pingTimer: number | undefined;
  let reconnectTimer: number | undefined;
  let closed = false;

  const open = () => {
    // A teardown (navigation/unmount) can race a scheduled reconnect — if we've
    // already been closed, drop it rather than spawning a zombie socket + ping.
    if (closed) return;
    const token = useAuthStore.getState().accessToken;
    // Browsers accept subprotocols made up of token characters only —
    // JWTs qualify (they're base64url-encoded).  We pass ["jwt", <jwt>]
    // so the server knows which subprotocol is the "name" and which is
    // the token payload.
    const protocols = token ? ["jwt", token] : undefined;
    ws = new WebSocket(url, protocols);

    ws.onopen = () => {
      retries = 0;
      pingTimer = window.setInterval(() => {
        if (ws?.readyState === 1) ws.send(JSON.stringify({ op: "ping" }));
      }, 30_000);
      opts.onOpen?.();
    };
    ws.onmessage = (ev) => {
      try {
        onMessage(JSON.parse(ev.data));
      } catch {
        /* ignore malformed frames */
      }
    };
    ws.onclose = (ev) => {
      window.clearInterval(pingTimer);
      opts.onClose?.(ev);
      if (closed) return;
      // Auth failures shouldn't trigger infinite reconnects — they need
      // a fresh token from the auth store.
      if (ev.code === 4401 || ev.code === 4403) {
        closed = true;
        return;
      }
      const backoff = Math.min(1000 * 2 ** retries++, 30_000);
      reconnectTimer = window.setTimeout(open, backoff);
    };
    ws.onerror = (e) => opts.onError?.(e);
  };
  open();

  return {
    send(msg: WSMessage) {
      if (ws?.readyState === 1) ws.send(JSON.stringify(msg));
    },
    close() {
      closed = true;
      window.clearTimeout(reconnectTimer);
      window.clearInterval(pingTimer);
      ws?.close();
    },
  };
}
