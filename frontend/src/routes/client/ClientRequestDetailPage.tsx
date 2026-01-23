import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { http } from "../../api/http";
import { CreditRequest } from "../../api/types";
import { AgentPanel } from "../../components/agents/AgentPanel";

export const ClientRequestDetailPage = () => {
  const { id } = useParams();
  const [data, setData] = useState<CreditRequest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await http.get<CreditRequest>(`/client/credit-requests/${id}`);
        setData(res);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    };
    if (id) load();
  }, [id]);

  if (loading) return <div className="card">Chargement...</div>;
  if (error) return <div className="card">Erreur: {error}</div>;
  if (!data) return <div className="card">Aucune donnée</div>;

  return (
    <div className="grid" style={{ gap: 12 }}>
      <div className="card">
        <h3>Statut</h3>
        <div className="badge">{data.status}</div>
        {data.customer_explanation && (
          <p style={{ marginTop: 12 }}>{String(data.customer_explanation)}</p>
        )}
      </div>
      <div className="card">
        <h3>Résumé</h3>
        <p style={{ color: "#475569" }}>{data.summary || "Résumé non disponible"}</p>
      </div>
      {data.agents?.explanation && (
        <AgentPanel title="Explication" agent={data.agents.explanation} />
      )}
    </div>
  );
};
