import { useEffect, useState, FormEvent } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { http } from "../../api/http";
import { AgentPanel } from "../../components/agents/AgentPanel";
import { BankerRequest, DecisionCreate } from "../../api/types";

export const BankerRequestDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState<BankerRequest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [decision, setDecision] = useState<DecisionCreate["decision"]>("approve");
  const [note, setNote] = useState("");
  const [comment, setComment] = useState("");

  const load = async () => {
    try {
      const res = await http.get<BankerRequest>(`/banker/credit-requests/${id}`);
      setData(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (id) load();
  }, [id]);

  const submitDecision = async (e: FormEvent) => {
    e.preventDefault();
    if (!id) return;
    try {
      await http.post(`/banker/credit-requests/${id}/decision`, { decision, note });
      setNote("");
      navigate("/banker/requests", { state: { toast: "Décision enregistrée." } });
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const submitComment = async (e: FormEvent) => {
    e.preventDefault();
    if (!id || !comment.trim()) return;
    try {
      await http.post(`/banker/credit-requests/${id}/comments`, { message: comment });
      setComment("");
      await load();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  if (loading) return <div className="card">Chargement...</div>;
  if (error) return <div className="card">Erreur: {error}</div>;
  if (!data) return <div className="card">Aucune donnée</div>;

  return (
    <div className="grid" style={{ gap: 12 }}>
      <div>
        <button className="button-ghost" type="button" onClick={() => navigate("/banker/requests")}>
          Retour à la liste
        </button>
      </div>
      <div className="card">
        <h2>Détail de la demande</h2>
        <div className="badge">{data.status}</div>
        <p style={{ color: "#475569", marginTop: 8 }}>
          Client {data.client_id} • {data.amount} € • {data.duration_months} mois
        </p>
        {data.summary && <p style={{ marginTop: 8 }}>{data.summary}</p>}
      </div>

      <div className="card">
        <h3>Profil financier</h3>
        <div style={{ display: "grid", gap: 6, color: "#475569" }}>
          <div>Revenus mensuels: {data.monthly_income} €</div>
          <div>Autres revenus: {data.other_income} €</div>
          <div>Charges mensuelles: {data.monthly_charges} €</div>
          <div>Type d'emploi: {data.employment_type}</div>
          <div>Contrat: {data.contract_type}</div>
          <div>Ancienneté: {data.seniority_years} ans</div>
          <div>Statut marital: {data.marital_status}</div>
          <div>Enfants: {data.number_of_children}</div>
          <div>Conjoint employé: {String(data.spouse_employed ?? "-")}</div>
          <div>Logement: {data.housing_status}</div>
          <div>Titulaire principal: {String(data.is_primary_holder ?? "-")}</div>
        </div>
      </div>

      <div className="card">
        <h3>Documents</h3>
        {data.documents && data.documents.length > 0 ? (
          <ul style={{ paddingLeft: 16 }}>
            {data.documents.map((doc) => (
              <li key={doc.document_id} style={{ color: "#475569" }}>
                {doc.document_type}: {doc.file_path}
              </li>
            ))}
          </ul>
        ) : (
          <p style={{ color: "#475569" }}>Aucun document.</p>
        )}
      </div>

      {data.agents?.document && <AgentPanel title="Agent Documents" agent={data.agents.document} />}
      {data.agents?.similarity && <AgentPanel title="Agent Similarité" agent={data.agents.similarity} />}
      {data.agents?.behavior && <AgentPanel title="Agent Comportement" agent={data.agents.behavior} />}
      {data.agents?.fraud && <AgentPanel title="Agent Fraude" agent={data.agents.fraud} />}

      <div className="card">
        <h3>Décision</h3>
        {data.decision ? (
          <>
            <div className="badge">{data.decision.decision}</div>
            {data.decision.note && <p style={{ marginTop: 8 }}>{data.decision.note}</p>}
            {data.decision.decided_at && (
              <p style={{ color: "#475569", marginTop: 8 }}>
                Décidé le {new Date(data.decision.decided_at).toLocaleString()}
              </p>
            )}
          </>
        ) : (
          <p style={{ color: "#475569" }}>Aucune décision pour le moment.</p>
        )}
        <form onSubmit={submitDecision} className="grid" style={{ gap: 8, marginTop: 12 }}>
          <label>Choisir une décision</label>
          <select className="input" value={decision} onChange={(e) => setDecision(e.target.value as DecisionCreate["decision"])}>
            <option value="approve">Approuver</option>
            <option value="reject">Refuser</option>
            <option value="review">Revoir</option>
          </select>
          <label>Note</label>
          <textarea className="input" value={note} onChange={(e) => setNote(e.target.value)} rows={3} />
          <button className="button-primary" type="submit">
            Enregistrer la décision
          </button>
        </form>
      </div>

      <div className="card">
        <h3>Commentaires</h3>
        {data.comments && data.comments.length > 0 ? (
          <ul style={{ paddingLeft: 16 }}>
            {data.comments.map((c, idx) => (
              <li key={`${c.created_at}-${idx}`} style={{ color: "#475569", marginBottom: 8 }}>
                <strong>{c.author_id}</strong>: {c.message}
              </li>
            ))}
          </ul>
        ) : (
          <p style={{ color: "#475569" }}>Aucun commentaire.</p>
        )}
        <form onSubmit={submitComment} style={{ marginTop: 12 }}>
          <label>Ajouter un commentaire</label>
          <textarea className="input" value={comment} onChange={(e) => setComment(e.target.value)} rows={3} />
          <button className="button-ghost" type="submit" style={{ marginTop: 8 }}>
            Ajouter le commentaire
          </button>
        </form>
      </div>
    </div>
  );
};
