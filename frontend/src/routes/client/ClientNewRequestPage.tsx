import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { http } from "../../api/http";
import { CreditRequestCreate, CreditRequest } from "../../api/types";

export const ClientNewRequestPage = () => {
  const navigate = useNavigate();
  const [amount, setAmount] = useState(5000);
  const [duration, setDuration] = useState(24);
  const [income, setIncome] = useState(3000);
  const [charges, setCharges] = useState(800);
  const [employment, setEmployment] = useState("employee");
  const [contract, setContract] = useState("permanent");
  const [seniority, setSeniority] = useState(5);
  const [family, setFamily] = useState("single");
  const [documents, setDocuments] = useState("salary.pdf, contract.pdf");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload: CreditRequestCreate = {
        amount,
        duration_months: duration,
        monthly_income: income,
        monthly_charges: charges,
        employment_type: employment,
        contract_type: contract,
        seniority_years: seniority,
        family_status: family,
        documents: documents.split(",").map((d) => d.trim()).filter(Boolean),
      };
      const res = await http.post<CreditRequest>("/client/credit-requests", payload);
      navigate(`/client/requests/${res.id}`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Demander un crédit</h2>
      <form className="grid" style={{ gap: 12 }} onSubmit={onSubmit}>
        <div className="grid two">
          <div className="form-group">
            <label>Montant (€)</label>
            <input className="input" type="number" value={amount} onChange={(e) => setAmount(Number(e.target.value))} required />
          </div>
          <div className="form-group">
            <label>Durée (mois)</label>
            <input className="input" type="number" value={duration} onChange={(e) => setDuration(Number(e.target.value))} required />
          </div>
        </div>
        <div className="grid two">
          <div className="form-group">
            <label>Revenus mensuels</label>
            <input className="input" type="number" value={income} onChange={(e) => setIncome(Number(e.target.value))} required />
          </div>
          <div className="form-group">
            <label>Charges mensuelles</label>
            <input className="input" type="number" value={charges} onChange={(e) => setCharges(Number(e.target.value))} required />
          </div>
        </div>
        <div className="grid two">
          <div className="form-group">
            <label>Emploi</label>
            <select className="input" value={employment} onChange={(e) => setEmployment(e.target.value)}>
              <option value="employee">Salarié</option>
              <option value="freelancer">Freelance</option>
              <option value="self_employed">Indépendant</option>
              <option value="unemployed">Sans emploi</option>
            </select>
          </div>
          <div className="form-group">
            <label>Contrat</label>
            <select className="input" value={contract} onChange={(e) => setContract(e.target.value)}>
              <option value="permanent">CDI</option>
              <option value="temporary">CDD</option>
              <option value="none">Aucun</option>
            </select>
          </div>
        </div>
        <div className="grid two">
          <div className="form-group">
            <label>Ancienneté (années)</label>
            <input className="input" type="number" value={seniority} onChange={(e) => setSeniority(Number(e.target.value))} />
          </div>
          <div className="form-group">
            <label>Situation familiale</label>
            <select className="input" value={family} onChange={(e) => setFamily(e.target.value)}>
              <option value="single">Célibataire</option>
              <option value="married">Marié</option>
            </select>
          </div>
        </div>
        <div className="form-group">
          <label>Documents (séparés par des virgules)</label>
          <input className="input" value={documents} onChange={(e) => setDocuments(e.target.value)} />
        </div>
        {error && <div style={{ color: "#b91c1c", fontSize: 14 }}>{error}</div>}
        <button className="button-primary" type="submit" disabled={loading}>
          {loading ? "Envoi..." : "Soumettre"}
        </button>
      </form>
    </div>
  );
};
