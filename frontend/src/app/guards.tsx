import { Navigate, useLocation } from "react-router-dom";
import { useAuthStore } from "@/stores/auth";

export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { accessToken } = useAuthStore();
  const loc = useLocation();
  if (!accessToken) return <Navigate to="/login" state={{ from: loc }} replace />;
  return <>{children}</>;
}
