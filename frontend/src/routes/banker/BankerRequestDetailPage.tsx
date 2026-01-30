import { useEffect, useState, FormEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { http } from "../../api/http";
import { AgentPanel } from "../../components/agents/AgentPanel";
import { AgentChatPanel } from "../../components/agents/AgentChatPanel";
import { BankerRequest, DecisionCreate } from "../../api/types";
import { formatCurrency, formatDate, formatDateTime, formatPercent, statusBadgeStyle, statusLabel } from "../../utils/format";
import { useAuthStore } from "../../features/auth/authStore";

const AGENTS = ["document", "behavior", "similarity", "fraud", "decision", "explanation", "final-decision"];
const AGENT_LABELS: Record<string, string> = {
  document: "Documents",
  behavior: "Comportement",
  similarity: "Similarité",
  fraud: "Fraude",
  decision: "Décision",
  explanation: "Explication",
  "final-decision": "Décision finale",
};

export const BankerRequestDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { token } = useAuthStore();
  const { logout } = useAuthStore();
  const [data, setData] = useState<BankerRequest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [decision, setDecision] = useState<DecisionCreate["decision"]>("approve");
  const [note, setNote] = useState("");
  const [selectedAgent, setSelectedAgent] = useState<string>("document");
  const [noteError, setNoteError] = useState<string | null>(null);
  const [suggesting, setSuggesting] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const load = async () => {
    try {
      const res = await http.get<BankerRequest>(`/banker/credit-requests/${id}`);
      setData(res);
      setLastUpdated(new Date().toISOString());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (id) load();
    const onFocus = () => {
      if (id) load();
    };
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [id]);

  const submitDecision = async (e: FormEvent) => {
    e.preventDefault();
    if (!id) return;
    if (decision === "reject" && !note.trim()) {
      setNoteError("La note est obligatoire en cas de refus.");
      return;
    }
    setNoteError(null);
    try {
      await http.post(`/banker/credit-requests/${id}/decision`, { decision, note });
      setNote("");
      navigate("/banker/requests", { state: { toast: "Décision enregistrée." } });
    } catch (err) {
      setError((err as Error).message);
    }
  };

  if (loading) return <div className="card">Chargement...</div>;
  if (error) return <div className="card">Erreur: {error}</div>;
  if (!data) return <div className="card">Aucune donnée</div>;

  const verdict = data.auto_decision || "review";
  const verdictLabel =
    verdict === "approve" ? "Vert (faible risque)" : verdict === "reject" ? "Rouge (risque élevé)" : "Orange (revue)";
  const verdictColor =
    verdict === "approve" ? "#16a34a" : verdict === "reject" ? "#dc2626" : "#f59e0b";

  const fetchSuggestedNote = async (mode: "reject" | "review") => {
    if (!id) return;
    setSuggesting(true);
    try {
      const res = await http.post<{ note: string }>(`/banker/credit-requests/${id}/decision-suggestion`, {
        decision: mode,
      });
      setNote(res.note || "");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSuggesting(false);
    }
  };

  const refresh = () => {
    setLoading(true);
    load();
  };

  const API_BASE = import.meta.env.VITE_API_URL || "/api";

  const fetchDocumentBlob = async (filename: string) => {
    const res = await fetch(`${API_BASE}/files/${data?.id}/${encodeURIComponent(filename)}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    return res.blob();
  };

  const handleViewDocument = async (filename: string) => {
    try {
      const blob = await fetchDocumentBlob(filename);
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank", "noopener");
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleDownloadDocument = async (filename: string) => {
    try {
      const blob = await fetchDocumentBlob(filename);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const selectedAgentData =
    data?.agents && selectedAgent && selectedAgent !== "final-decision"
      ? (data.agents as Record<string, typeof data.agents.document | undefined>)[selectedAgent]
      : undefined;

  const installments = data.installments || [];
  const payments = data.payments || [];
  const paymentSummary = data.payment_behavior_summary;

  const showDecisionForm = !data.decision || data.decision.decision === "review";

  return (
    <div className="banker-detail">
      <div className="detail-topbar">
        <Link to="/banker/requests" className="detail-back">
          ← Retour aux demandes
        </Link>
        <div className="detail-topbar-actions">
          <span>{lastUpdated ? `Dernière mise à jour: ${formatDateTime(lastUpdated)}` : "Dernière mise à jour: —"}</span>
          <button className="button-ghost" type="button" onClick={refresh} disabled={loading}>
            {loading ? "Mise à jour..." : "Rafraîchir"}
          </button>
          <button
            className="button-ghost"
            type="button"
            onClick={() => {
              logout();
              navigate("/login");
            }}
          >
            Déconnexion
          </button>
        </div>
      </div>

      <div className="detail-hero">
        <div>
          <div className="detail-title-row">
            <h1>Demande #{data.id}</h1>
            <span className="badge" style={statusBadgeStyle(data.status)}>
              {statusLabel(data.status)}
            </span>
            {data.auto_review_required && <span className="detail-pill">Revue requise</span>}
          </div>
          <p className="detail-sub">
            Client {data.client_id} • Créée le {formatDateTime(data.created_at)} • Mise à jour le{" "}
            {formatDateTime(data.updated_at)}
          </p>
          {data.summary && <p className="detail-summary">{data.summary}</p>}
        </div>
        <div className="detail-verdict">
          <div className="detail-verdict-title">Verdict automatique</div>
          <div
            className="badge"
            style={{ background: verdictColor + "22", color: verdictColor, border: "1px solid " + verdictColor + "55" }}
          >
            {verdictLabel}
          </div>
          {typeof data.auto_decision_confidence === "number" && (
            <div className="detail-verdict-sub">Confiance: {(data.auto_decision_confidence * 100).toFixed(0)}%</div>
          )}
        </div>
      </div>

      <div className="detail-kpis">
        <div className="detail-kpi-card">
          <span>Montant demandé</span>
          <strong>{formatCurrency(data.amount)} €</strong>
        </div>
        <div className="detail-kpi-card">
          <span>Durée</span>
          <strong>{data.duration_months ?? "—"} mois</strong>
        </div>
        <div className="detail-kpi-card">
          <span>Revenus mensuels</span>
          <strong>{formatCurrency(data.monthly_income)} €</strong>
        </div>
        <div className="detail-kpi-card">
          <span>Charges mensuelles</span>
          <strong>{formatCurrency(data.monthly_charges)} €</strong>
        </div>
      </div>

      <div className="detail-grid">
        <div className="card detail-card">
          <h3>Profil financier</h3>
          <div className="detail-list">
            <div>Revenus mensuels: {formatCurrency(data.monthly_income)} €</div>
            <div>Autres revenus: {formatCurrency(data.other_income)} €</div>
            <div>Charges mensuelles: {formatCurrency(data.monthly_charges)} €</div>
            <div>Type d'emploi: {data.employment_type ?? "—"}</div>
            <div>Contrat: {data.contract_type ?? "—"}</div>
            <div>Ancienneté: {data.seniority_years ?? "—"} ans</div>
            <div>Statut marital: {data.marital_status ?? "—"}</div>
            <div>Enfants: {data.number_of_children ?? "—"}</div>
            <div>Conjoint employé: {String(data.spouse_employed ?? "-")}</div>
            <div>Logement: {data.housing_status ?? "—"}</div>
            <div>Titulaire principal: {String(data.is_primary_holder ?? "-")}</div>
          </div>
        </div>

        <div className="card detail-card">
          <h3>Documents</h3>
          {data.documents && data.documents.length > 0 ? (
            <ul className="detail-docs">
              {data.documents.map((doc) => {
                const filename = doc.file_path.split("/").pop() || doc.file_path;
                return (
                  <li key={doc.document_id}>
                    <div className="detail-docs-meta">
                      <span className="detail-docs-name">{filename}</span>
                      <span className="detail-docs-type">{doc.document_type}</span>
                    </div>
                    <div className="detail-docs-actions">
                      <button
                        type="button"
                        className="detail-docs-button"
                        onClick={() => handleViewDocument(filename)}
                      >
                        Voir
                      </button>
                      <button
                        type="button"
                        className="detail-docs-button ghost"
                        onClick={() => handleDownloadDocument(filename)}
                      >
                        Télécharger
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="detail-muted">Aucun document.</p>
          )}
        </div>

        {data.loan && (
          <div className="card detail-card">
            <h3>Prêt réel</h3>
            <div className="detail-list">
              <div>Montant: {formatCurrency(data.loan.principal_amount)} €</div>
              <div>Taux: {(data.loan.interest_rate * 100).toFixed(2)}%</div>
              <div>Durée: {data.loan.term_months} mois</div>
              <div>Statut: {data.loan.status}</div>
              <div>Début: {formatDate(data.loan.start_date)}</div>
              <div>Fin: {formatDate(data.loan.end_date)}</div>
            </div>
          </div>
        )}

        {paymentSummary && (
          <div className="card detail-card">
            <h3>Résumé comportement de paiement</h3>
            <div className="detail-list">
              <div>Taux à l'heure: {formatPercent(paymentSummary.on_time_rate)}</div>
              <div>Retard moyen: {paymentSummary.avg_days_late?.toFixed(1)} jours</div>
              <div>Retard max: {paymentSummary.max_days_late} jours</div>
              <div>Tranches manquées: {paymentSummary.missed_installments}</div>
              <div>Dernier paiement: {formatDate(paymentSummary.last_payment_date)}</div>
            </div>
          </div>
        )}

        {installments.length > 0 && (
          <div className="card detail-card">
            <h3>Tranches de paiement</h3>
            <ul className="detail-list">
              {installments.slice(0, 12).map((inst) => (
                <li key={inst.installment_id}>
                  #{inst.installment_number} • {formatDate(inst.due_date)} • {formatCurrency(inst.amount_due)} € •{" "}
                  {inst.status}
                  {typeof inst.days_late === "number" && inst.days_late > 0
                    ? ` (retard ${inst.days_late}j)`
                    : ""}
                  {inst.amount_paid ? ` • payé ${formatCurrency(inst.amount_paid)} €` : ""}
                </li>
              ))}
            </ul>
            {installments.length > 12 && (
              <p className="detail-muted">{installments.length - 12} tranches supplémentaires…</p>
            )}
          </div>
        )}

        {payments.length > 0 && (
          <div className="card detail-card">
            <h3>Paiements réels</h3>
            <ul className="detail-list">
              {payments.slice(0, 12).map((pay) => (
                <li key={pay.payment_id}>
                  {formatDate(pay.payment_date)} • {formatCurrency(pay.amount)} € • {pay.channel} • {pay.status}
                  {pay.is_reversal ? " (reversal)" : ""}
                </li>
              ))}
            </ul>
            {payments.length > 12 && (
              <p className="detail-muted">{payments.length - 12} paiements supplémentaires…</p>
            )}
          </div>
        )}

      </div>

      <div className="detail-section">
        <div className="detail-section-header">
          <h2>Agents</h2>
          <p>Ouvrez un agent pour consulter l'analyse et discuter.</p>
        </div>
        <div className="detail-tabs">
          {AGENTS.map((agent) => (
            <button
              key={agent}
              type="button"
              className={`detail-tab ${selectedAgent === agent ? "active" : ""}`}
              onClick={() => setSelectedAgent(agent)}
            >
              {AGENT_LABELS[agent] ?? agent}
            </button>
          ))}
        </div>
        <div className="detail-agent-grid">
          <div className="detail-agent-pane">
            {selectedAgent === "final-decision" ? (
              <div className="card detail-card">
                <h3>Décision finale</h3>
                {data.decision ? (
                  <>
                    <div className="badge">{data.decision.decision}</div>
                    {data.decision.note && <p style={{ marginTop: 8 }}>{data.decision.note}</p>}
                    {data.decision.decided_at && (
                      <p className="detail-muted">Décidé le {new Date(data.decision.decided_at).toLocaleString()}</p>
                    )}
                  </>
                ) : (
                  <p className="detail-muted">Aucune décision pour le moment.</p>
                )}
                {showDecisionForm && (
                  <form onSubmit={submitDecision} className="grid" style={{ gap: 8, marginTop: 12 }}>
                    <label>Choisir une décision</label>
                    <select
                      className="input"
                      value={decision}
                      onChange={(e) => {
                        setDecision(e.target.value as DecisionCreate["decision"]);
                        setNoteError(null);
                      }}
                    >
                      <option value="approve">Approuver</option>
                      <option value="reject">Refuser</option>
                      <option value="review">Revoir</option>
                    </select>
                    <label>Note</label>
                    <textarea className="input" value={note} onChange={(e) => setNote(e.target.value)} rows={3} />
                    {noteError && <div style={{ color: "#b91c1c", fontSize: 14 }}>{noteError}</div>}
                    {decision === "reject" && (
                      <button
                        className="button-ghost"
                        type="button"
                        onClick={() => fetchSuggestedNote("reject")}
                        style={{ justifySelf: "start" }}
                        disabled={suggesting}
                      >
                        {suggesting ? "Suggestion..." : "Suggérer une cause (AI)"}
                      </button>
                    )}
                    {decision === "review" && (
                      <button
                        className="button-ghost"
                        type="button"
                        onClick={() => fetchSuggestedNote("review")}
                        style={{ justifySelf: "start" }}
                        disabled={suggesting}
                      >
                        {suggesting ? "Suggestion..." : "Suggérer des actions (AI)"}
                      </button>
                    )}
                    <button className="button-primary" type="submit">
                      Enregistrer la décision
                    </button>
                  </form>
                )}
              </div>
            ) : selectedAgentData ? (
              <AgentPanel title={`Agent ${AGENT_LABELS[selectedAgent] ?? selectedAgent}`} agent={selectedAgentData} />
            ) : (
              <div className="card detail-card">
                <h3>Analyse indisponible</h3>
                <p className="detail-muted">Aucune donnée pour cet agent pour le moment.</p>
              </div>
            )}
          </div>
          <div className="detail-agent-pane">
            {selectedAgent === "final-decision" ? null : id && <AgentChatPanel requestId={id} agentName={selectedAgent} />}
          </div>
        </div>
      </div>
    </div>
  );
};
