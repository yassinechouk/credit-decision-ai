import { AgentResult } from "../../api/types";

interface Props {
  title: string;
  agent: AgentResult;
}

export const AgentPanel = ({ title, agent }: Props) => {
  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        {typeof agent.score === "number" && (
          <span className="badge">Score: {agent.score.toFixed(2)}</span>
        )}
        {typeof agent.confidence === "number" && (
          <span className="badge" style={{ background: "#dbeafe", color: "#1e3a8a" }}>
            Confiance: {(agent.confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>
      {agent.flags && agent.flags.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <strong>Signaux:</strong>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6 }}>
            {agent.flags.map((flag) => (
              <span key={flag} className="badge" style={{ background: "#fef3c7", color: "#92400e" }}>
                {flag}
              </span>
            ))}
          </div>
        </div>
      )}
      {agent.explanations?.global_summary && (
        <p style={{ marginTop: 8, color: "#475569" }}>{agent.explanations.global_summary}</p>
      )}
      {agent.explanations?.flag_explanations && (
        <div style={{ marginTop: 8 }}>
          <strong>Détails:</strong>
          <ul style={{ paddingLeft: 16 }}>
            {Object.entries(agent.explanations.flag_explanations).map(([flag, desc]) => (
              <li key={flag} style={{ color: "#475569" }}>
                <strong>{flag}:</strong> {desc}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
