import { Navigate } from "react-router-dom";
import { ReactNode } from "react";
import { useAuthStore, UserRole } from "../../features/auth/authStore";

export const RoleGuard = ({ allow, children }: { allow: Exclude<UserRole, null>[]; children: ReactNode }) => {
  const { role } = useAuthStore();
  if (!role || !allow.includes(role)) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
};
