import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  tenantId: string | null;
  userEmail: string | null;
  setTokens: (access: string, refresh: string) => void;
  setTenant: (tenantId: string) => void;
  setEmail: (email: string) => void;
  clear: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      accessToken: null,
      refreshToken: null,
      tenantId: null,
      userEmail: null,
      setTokens: (access, refresh) => set({ accessToken: access, refreshToken: refresh }),
      setTenant: (tenantId) => set({ tenantId }),
      setEmail: (userEmail) => set({ userEmail }),
      clear: () => set({ accessToken: null, refreshToken: null, tenantId: null, userEmail: null }),
    }),
    { name: "alphadesk-auth" },
  ),
);
