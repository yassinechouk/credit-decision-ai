import { useEffect, useState } from "react";
import { http } from "../../api/http";
import { AgentChatResponse, AgentChatMessage } from "../../api/types";

interface Props {
  requestId: string;
  agentName: string;
}

export const AgentChatPanel = ({ requestId, agentName }: Props) => {
  const [messages, setMessages] = useState<AgentChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadMessages = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await http.get<AgentChatResponse>(`/banker/credit-requests/${requestId}/agent-chat/${agentName}`);
      setMessages(res.messages || []);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (requestId && agentName) {
      loadMessages();
    }
  }, [requestId, agentName]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await http.post<AgentChatResponse>(`/banker/credit-requests/${requestId}/agent-chat`, {
        agent_name: agentName,
        message: input.trim(),
      });
      setMessages(res.messages || []);
      setInput("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h3>Discussion avec l’agent {agentName}</h3>
      {error && <div style={{ color: "#b91c1c", fontSize: 14 }}>{error}</div>}
      <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 280, overflow: "auto", marginTop: 8 }}>
        {messages.length === 0 && <div style={{ color: "#64748b" }}>Aucun message pour le moment.</div>}
        {messages.map((msg, idx) => (
          <div
            key={`${msg.created_at}-${idx}`}
            style={{
              alignSelf: msg.role === "banker" ? "flex-end" : "flex-start",
              background: msg.role === "banker" ? "#e2e8f0" : "#fef3c7",
              color: "#0f172a",
              padding: "8px 10px",
              borderRadius: 10,
              maxWidth: "80%",
              fontSize: 14,
            }}
          >
            <div style={{ whiteSpace: "pre-wrap" }}>{msg.content}</div>
            {msg.structured_output && msg.role === "agent" && (
              <div style={{ marginTop: 6 }}>
                <details>
                  <summary style={{ cursor: "pointer", color: "#64748b" }}>Voir données</summary>
                  <pre style={{ whiteSpace: "pre-wrap", fontSize: 12, color: "#475569" }}>
                    {JSON.stringify(msg.structured_output, null, 2)}
                  </pre>
                </details>
              </div>
            )}
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <input
          className="input"
          placeholder="Pose ta question…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
        />
        <button className="button-primary" type="button" onClick={sendMessage} disabled={loading}>
          {loading ? "Envoi..." : "Envoyer"}
        </button>
      </div>
    </div>
  );
};
