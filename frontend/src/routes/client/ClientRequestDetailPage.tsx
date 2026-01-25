import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { http } from "../../api/http";
import { CreditRequest } from "../../api/types";
import { AgentPanel } from "../../components/agents/AgentPanel";

export const ClientRequestDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
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

  const verdict = data.auto_decision || "review";
  const verdictLabel =
    verdict === "approve" ? "Vert (faible risque)" : verdict === "reject" ? "Rouge (risque élevé)" : "Orange (revue)";
  const verdictColor =
    verdict === "approve" ? "#16a34a" : verdict === "reject" ? "#dc2626" : "#f59e0b";

  return (
    <div className="grid" style={{ gap: 12 }}>
      <div>
        <button className="button-ghost" type="button" onClick={() => navigate("/client/requests")}>
          Retour à mes demandes
        </button>
      </div>
      <div className="card">
        <h3>Verdict automatique</h3>
        <div
          className="badge"
          style={{ background: verdictColor + "22", color: verdictColor, border: "1px solid " + verdictColor + "55" }}
        >
          {verdictLabel}
        </div>
        {typeof data.auto_decision_confidence === "number" && (
          <p style={{ color: "#475569", marginTop: 8 }}>
            Confiance: {(data.auto_decision_confidence * 100).toFixed(0)}%
          </p>
        )}
        {data.auto_review_required && (
          <p style={{ marginTop: 6, color: "#b45309" }}>Revue humaine requise.</p>
        )}
      </div>
      <div className="card">
        <h3>Statut</h3>
        <div className="badge">{data.status}</div>
        {data.customer_explanation && (
          <p style={{ marginTop: 12 }}>{String(data.customer_explanation)}</p>
        )}
      </div>
      {data.decision && (
        <div className="card">
          <h3>Décision</h3>
          <div className="badge">{data.decision.decision}</div>
          {data.decision.note && <p style={{ marginTop: 8 }}>{data.decision.note}</p>}
          {data.decision.decided_at && (
            <p style={{ color: "#475569", marginTop: 8 }}>
              Décidé le {new Date(data.decision.decided_at).toLocaleString()}
            </p>
          )}
        </div>
      )}
      <div className="card">
        <h3>Résumé</h3>
        <p style={{ color: "#475569" }}>{data.summary || "Résumé non disponible"}</p>
      </div>
      {data.comments && data.comments.length > 0 && (
        <div className="card">
          <h3>Commentaires du banquier</h3>
          <ul style={{ paddingLeft: 16 }}>
            {data.comments.map((comment, idx) => (
              <li key={`${comment.created_at}-${idx}`} style={{ marginBottom: 8, color: "#475569" }}>
                <strong>{comment.author_id}</strong>: {comment.message}
              </li>
            ))}
          </ul>
        </div>
      )}
      {data.agents?.explanation && (
        <AgentPanel title="Explication" agent={data.agents.explanation} />
      )}
    </div>
  );
};
