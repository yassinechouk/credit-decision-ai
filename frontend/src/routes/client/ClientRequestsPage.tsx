import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { http } from "../../api/http";
import { CreditRequest } from "../../api/types";

export const ClientRequestsPage = () => {
  const [items, setItems] = useState<CreditRequest[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await http.get<CreditRequest[]>("/client/credit-requests");
        setItems(res);
      } catch (err) {
        setError((err as Error).message);
      }
    };
    load();
  }, []);

  if (error) return <div className="card">Erreur: {error}</div>;

  return (
    <div className="grid" style={{ gap: 12 }}>
      <div className="card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2>Mes demandes</h2>
          <p style={{ color: "#475569" }}>Suivez l'état de vos demandes de crédit.</p>
        </div>
        <Link className="button-primary" to="/client/requests/new">
          Nouvelle demande
        </Link>
      </div>
      {items.length === 0 ? (
        <div className="card">
          <p style={{ color: "#475569" }}>Aucune demande pour le moment.</p>
        </div>
      ) : (
        items.map((req) => (
          <div key={req.id} className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <strong>Demande #{req.id}</strong>
                <div style={{ color: "#475569", fontSize: 14 }}>
                  Créée le {new Date(req.created_at).toLocaleString()}
                </div>
              </div>
              <span className="badge">{req.status}</span>
            </div>
            {req.summary && <p style={{ marginTop: 8, color: "#475569" }}>{req.summary}</p>}
            <div style={{ marginTop: 8 }}>
              <Link className="button-ghost" to={`/client/requests/${req.id}`}>
                Voir détail
              </Link>
            </div>
          </div>
        ))
      )}
    </div>
  );
};
