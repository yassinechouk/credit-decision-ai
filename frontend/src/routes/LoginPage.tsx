import { FormEvent, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuthStore } from "../features/auth/authStore";
import { http } from "../api/http";
import { LoginRequest, LoginResponse } from "../api/types";

export const LoginPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { setAuth } = useAuthStore();
  const [email, setEmail] = useState("banker@example.com");
  const [password, setPassword] = useState("secret");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload: LoginRequest = { email, password };
      const res = await http.post<LoginResponse>('/auth/login', payload, { auth: false });
      setAuth({ token: res.token, role: res.role, userId: res.user_id });
      if (res.role === "banker") navigate("/banker/requests", { replace: true });
      else navigate("/client/requests/new", { replace: true, state: location.state });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card" style={{ maxWidth: 420, margin: "40px auto" }}>
      <h2>Connexion</h2>
      <p style={{ color: "#475569" }}>Identifiez-vous pour accéder à votre espace.</p>
      <form className="grid" style={{ gap: 12 }} onSubmit={onSubmit}>
        <div className="form-group">
          <label>Email</label>
          <input className="input" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </div>
        <div className="form-group">
          <label>Mot de passe</label>
          <input className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </div>
        {error && <div style={{ color: "#b91c1c", fontSize: 14 }}>{error}</div>}
        <button className="button-primary" type="submit" disabled={loading}>
          {loading ? "Connexion..." : "Se connecter"}
        </button>
      </form>
    </div>
  );
};
