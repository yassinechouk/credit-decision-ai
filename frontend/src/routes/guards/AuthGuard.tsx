import { Navigate, useLocation } from "react-router-dom";
import { useAuthStore } from "../../features/auth/authStore";
import { ReactNode } from "react";

export const AuthGuard = ({ children }: { children: ReactNode }) => {
  const { token } = useAuthStore();
  const location = useLocation();
  if (!token) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }
  return <>{children}</>;
};
