import axios, { AxiosError, AxiosInstance, AxiosRequestConfig, AxiosResponse } from "axios";
import { useAuthStore } from "@/stores/auth";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "/api/v1",
  timeout: 30_000,
});

/** Attach the current JWT to every outbound request on the given client. */
function attachAuth(client: AxiosInstance) {
  client.interceptors.request.use((cfg) => {
    const t = useAuthStore.getState().accessToken;
    if (t) {
      cfg.headers = cfg.headers ?? {};
      cfg.headers["Authorization"] = `Bearer ${t}`;
    }
    return cfg;
  });
}

attachAuth(api);

let refreshing: Promise<string | null> | null = null;

// ---------------------------------------------------------------------------
// DRF CursorPagination envelope auto-unwrap.
//
// Every ModelViewSet in the backend is globally paginated — GET /foo/ replies
// with `{ next, previous, results: [...] }` instead of a bare array.  The
// frontend consistently types list endpoints as `T[]` and renders them with
// `<DataTable rows={data}/>`; feeding the envelope object into a component
// that calls `rows.length` / `rows.map(...)` was crashing every nav page.
//
// We unwrap here, in one place, whenever the body looks like the paginated
// envelope.  The shape check is strict so nothing else trips it: presence of
// `results` (array) AND both `next` / `previous` fields.
// ---------------------------------------------------------------------------
function isPaginatedEnvelope(body: unknown): body is { results: unknown[] } {
  return (
    typeof body === "object" &&
    body !== null &&
    Array.isArray((body as { results?: unknown }).results) &&
    "next" in body &&
    "previous" in body
  );
}

/** Request config can opt out of the auto-unwrap to keep the cursor links
 *  (needed for "load more" pagination — see `fetchPage`). */
type EnvelopeConfig = AxiosRequestConfig & { _envelope?: boolean };

/** Shared response interceptor — unwraps pagination + handles 401 refresh. */
function attachResponse(client: AxiosInstance) {
  client.interceptors.response.use(
    (r: AxiosResponse) => {
      if (!(r.config as EnvelopeConfig)?._envelope && isPaginatedEnvelope(r.data)) {
        r.data = r.data.results;
      }
      return r;
    },
    async (err: AxiosError) => {
      const original = err.config as AxiosRequestConfig & { _retried?: boolean };
      if (err.response?.status === 401 && !original._retried) {
        original._retried = true;
        if (!refreshing) refreshing = refresh();
        const newToken = await refreshing;
        refreshing = null;
        if (newToken) {
          original.headers = { ...(original.headers ?? {}), Authorization: `Bearer ${newToken}` };
          return client.request(original);
        }
      }
      return Promise.reject(err);
    },
  );
}

attachResponse(api);

// ---------------------------------------------------------------------------
// `list<T>(url)` — tiny helper for list GETs that should always hand back an
// array, even when a transient non-envelope response comes through (server
// error body, empty 204, paginator turned off, etc.).  Falls back to `[]` so
// downstream `.map(...)` / `rows.length` never crash the page.
// ---------------------------------------------------------------------------
/** One cursor-paginated page (DRF CursorPagination envelope, links preserved). */
export interface Page<T> {
  results: T[];
  next: string | null;
  previous: string | null;
}

/** Fetch a cursor-paginated page WITHOUT unwrapping, so callers can follow
 *  `next` for infinite scroll / "load more". Pass a relative path for page 1,
 *  or the absolute `next` URL returned by the previous page. */
export async function fetchPage<T>(url: string, params?: Record<string, unknown>): Promise<Page<T>> {
  const { data } = await api.get<Page<T>>(url, { _envelope: true, params } as EnvelopeConfig);
  return {
    results: Array.isArray(data?.results) ? data.results : [],
    next: data?.next ?? null,
    previous: data?.previous ?? null,
  };
}

export async function list<T>(url: string, config?: AxiosRequestConfig): Promise<T[]> {
  const { data } = await api.get<T[] | { results: T[] } | null | undefined>(url, config);
  if (Array.isArray(data)) return data;
  if (data && typeof data === "object" && Array.isArray((data as { results?: T[] }).results)) {
    return (data as { results: T[] }).results;
  }
  return [];
}

async function refresh(): Promise<string | null> {
  const refresh_token = useAuthStore.getState().refreshToken;
  if (!refresh_token) return null;
  try {
    const { data } = await axios.post(`${api.defaults.baseURL}/auth/token/refresh/`, { refresh: refresh_token });
    useAuthStore.getState().setTokens(data.access, data.refresh ?? refresh_token);
    return data.access;
  } catch {
    useAuthStore.getState().clear();
    return null;
  }
}
