import { useAuthStore } from "../features/auth/authStore";

const API_BASE = import.meta.env.VITE_API_URL || "/api";

type HttpOptions = RequestInit & { auth?: boolean };

type HttpClient = {
  get<T>(path: string, opts?: HttpOptions): Promise<T>;
  post<T>(path: string, body?: unknown, opts?: HttpOptions): Promise<T>;
};

const makeHeaders = (auth: boolean): HeadersInit => {
  const headers: HeadersInit = { "Content-Type": "application/json" };
  const token = useAuthStore.getState().token;
  if (auth && token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
};

const handle = async <T>(res: Response): Promise<T> => {
  if (res.status === 204) return undefined as unknown as T;
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
};

export const http: HttpClient = {
  async get<T>(path: string, opts: HttpOptions = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
      ...opts,
      headers: { ...makeHeaders(opts.auth ?? true), ...(opts.headers || {}) },
    });
    return handle<T>(res);
  },
  async post<T>(path: string, body?: unknown, opts: HttpOptions = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      body: body ? JSON.stringify(body) : undefined,
      ...opts,
      headers: { ...makeHeaders(opts.auth ?? true), ...(opts.headers || {}) },
    });
    return handle<T>(res);
  },
};
