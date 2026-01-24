import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { http } from "../../api/http";
import { BankerRequest } from "../../api/types";

const RequestsSection = ({ title, items }: { title: string; items: BankerRequest[] }) => (
  <div className="card">
    <h2>{title}</h2>
    {items.length === 0 ? (
      <p style={{ color: "#475569" }}>Aucune demande.</p>
    ) : (
      <div className="grid" style={{ gap: 12 }}>
        {items.map((req) => (
          <div key={req.id} className="card" style={{ border: "1px solid #e2e8f0" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <strong>Demande #{req.id}</strong>
                <div style={{ color: "#475569", fontSize: 14 }}>
                  Client {req.client_id} • {req.amount} € • {req.duration_months} mois
                </div>
              </div>
              <span className="badge">{req.status}</span>
            </div>
            <div style={{ marginTop: 8, display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#475569" }}>
                Créée le {new Date(req.created_at).toLocaleString()}
              </span>
              <Link className="button-ghost" to={`/banker/requests/${req.id}`}>
                Voir détail
              </Link>
            </div>
          </div>
        ))}
      </div>
    )}
  </div>
);

export const BankerRequestsPage = () => {
  const [pending, setPending] = useState<BankerRequest[]>([]);
  const [decided, setDecided] = useState<BankerRequest[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const location = useLocation();

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const [pendingRes, decidedRes] = await Promise.all([
          http.get<BankerRequest[]>("/banker/credit-requests?status=pending"),
          http.get<BankerRequest[]>("/banker/credit-requests?status=decided"),
        ]);
        if (!active) return;
        setPending(pendingRes);
        setDecided(decidedRes);
      } catch (err) {
        if (!active) return;
        setError((err as Error).message);
      }
    };
    load();
    const interval = setInterval(load, 15000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const state = location.state as { toast?: string } | null;
    if (state?.toast) {
      setToast(state.toast);
      const timer = setTimeout(() => setToast(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [location.state]);

  if (error) return <div className="card">Erreur: {error}</div>;

  return (
    <div className="grid" style={{ gap: 16 }}>
      {toast && (
        <div className="card" style={{ background: "#ecfeff", borderColor: "#99f6e4", color: "#0f766e" }}>
          {toast}
        </div>
      )}
      <RequestsSection title="Demandes à traiter" items={pending} />
      <RequestsSection title="Demandes traitées" items={decided} />
    </div>
  );
};
