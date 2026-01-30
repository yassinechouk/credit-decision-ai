import { create } from "zustand";

export type UserRole = "client" | "banker" | null;

interface AuthState {
  token: string | null;
  role: UserRole;
  userId: string | null;
  setAuth: (data: { token: string; role: Exclude<UserRole, null>; userId: string }) => void;
  logout: () => void;
}

const STORAGE_KEY = "cdai_auth";
const API_BASE = import.meta.env.VITE_API_URL || "/api";

const load = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as { token: string; role: UserRole; userId: string };
  } catch {
    return null;
  }
};

export const useAuthStore = create<AuthState>((set, get) => ({
  token: load()?.token ?? null,
  role: load()?.role ?? null,
  userId: load()?.userId ?? null,
  setAuth: (data) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    set({ token: data.token, role: data.role, userId: data.userId });
  },
  logout: () => {
    const token = get().token;
    if (token) {
      fetch(`${API_BASE}/auth/logout`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      }).catch(() => {});
    }
    localStorage.removeItem(STORAGE_KEY);
    set({ token: null, role: null, userId: null });
  },
}));
