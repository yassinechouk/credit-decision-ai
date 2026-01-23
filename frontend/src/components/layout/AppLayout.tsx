import { Link, Outlet, useNavigate } from "react-router-dom";
import { useAuthStore } from "../../features/auth/authStore";

export const AppLayout = () => {
  const navigate = useNavigate();
  const { token, role, logout } = useAuthStore();

  return (
    <div className="layout-shell">
      <header className="navbar">
        <div className="brand">Credit Decision AI</div>
        <div className="actions">
          {role === "client" && <Link to="/client/requests/new">Demander un crédit</Link>}
          {role === "banker" && <Link to="/banker/requests">Demandes</Link>}
          {token ? (
            <button className="button-ghost" onClick={() => { logout(); navigate("/login"); }}>
              Déconnexion
            </button>
          ) : (
            <Link to="/login">Connexion</Link>
          )}
        </div>
      </header>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
};
