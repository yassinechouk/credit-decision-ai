import { FormEvent, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { http } from "../../api/http";
import { CreditRequestCreate, CreditRequest } from "../../api/types";

export const ClientNewRequestPage = () => {
  const navigate = useNavigate();
  const [amount, setAmount] = useState(5000);
  const [duration, setDuration] = useState(24);
  const [income, setIncome] = useState(3000);
  const [otherIncome, setOtherIncome] = useState(0);
  const [charges, setCharges] = useState(800);
  const [employment, setEmployment] = useState("employee");
  const [contract, setContract] = useState("permanent");
  const [seniority, setSeniority] = useState(5);
  const [family, setFamily] = useState("single");
  const [childrenCount, setChildrenCount] = useState(0);
  const [spouseEmployed, setSpouseEmployed] = useState<"unknown" | "yes" | "no">("unknown");
  const [housingStatus, setHousingStatus] = useState("owner");
  const [isPrimaryHolder, setIsPrimaryHolder] = useState(true);
  const [documents, setDocuments] = useState<File[]>([]);
  const [documentNames, setDocumentNames] = useState("salary.pdf, contract.pdf");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const startRef = useRef(Date.now());
  const editCountRef = useRef(0);
  const incomeEditRef = useRef(0);
  const docReuploadRef = useRef(0);

  const markEdit = (type?: "income") => {
    editCountRef.current += 1;
    if (type === "income") incomeEditRef.current += 1;
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const submissionSeconds = Math.max(0, Math.round((Date.now() - startRef.current) / 1000));
      const telemetry = {
        submission_duration_seconds: submissionSeconds,
        number_of_edits: editCountRef.current,
        income_field_edits: incomeEditRef.current,
        document_reuploads: docReuploadRef.current,
        back_navigation_count: 0,
        form_abandon_attempts: 0,
      };
      const payload: CreditRequestCreate = {
        amount,
        duration_months: duration,
        monthly_income: income,
        other_income: otherIncome,
        monthly_charges: charges,
        employment_type: employment,
        contract_type: contract,
        seniority_years: seniority,
        family_status: family,
        number_of_children: childrenCount,
        spouse_employed: spouseEmployed === "unknown" ? undefined : spouseEmployed === "yes",
        housing_status: housingStatus,
        is_primary_holder: isPrimaryHolder,
        documents: documentNames.split(",").map((d) => d.trim()).filter(Boolean),
        telemetry,
      };
      let res: CreditRequest;
      if (documents.length > 0) {
        const form = new FormData();
        form.append("payload", JSON.stringify(payload));
        documents.forEach((file) => form.append("files", file));
        res = await http.postForm<CreditRequest>("/client/credit-requests/upload", form);
      } else {
        res = await http.post<CreditRequest>("/client/credit-requests", payload);
      }
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
            <input
              className="input"
              type="number"
              value={amount}
              onChange={(e) => {
                markEdit();
                setAmount(Number(e.target.value));
              }}
              required
            />
          </div>
          <div className="form-group">
            <label>Durée (mois)</label>
            <input
              className="input"
              type="number"
              value={duration}
              onChange={(e) => {
                markEdit();
                setDuration(Number(e.target.value));
              }}
              required
            />
          </div>
        </div>
        <div className="grid two">
          <div className="form-group">
            <label>Revenus mensuels</label>
            <input
              className="input"
              type="number"
              value={income}
              onChange={(e) => {
                markEdit("income");
                setIncome(Number(e.target.value));
              }}
              required
            />
          </div>
          <div className="form-group">
            <label>Charges mensuelles</label>
            <input
              className="input"
              type="number"
              value={charges}
              onChange={(e) => {
                markEdit();
                setCharges(Number(e.target.value));
              }}
              required
            />
          </div>
        </div>
        <div className="grid two">
          <div className="form-group">
            <label>Autres revenus</label>
            <input
              className="input"
              type="number"
              value={otherIncome}
              onChange={(e) => {
                markEdit("income");
                setOtherIncome(Number(e.target.value));
              }}
            />
          </div>
          <div className="form-group">
            <label>Nombre d'enfants</label>
            <input
              className="input"
              type="number"
              value={childrenCount}
              onChange={(e) => {
                markEdit();
                setChildrenCount(Number(e.target.value));
              }}
            />
          </div>
        </div>
        <div className="grid two">
          <div className="form-group">
            <label>Emploi</label>
            <select
              className="input"
              value={employment}
              onChange={(e) => {
                markEdit();
                setEmployment(e.target.value);
              }}
            >
              <option value="employee">Salarié</option>
              <option value="freelancer">Freelance</option>
              <option value="self_employed">Indépendant</option>
              <option value="unemployed">Sans emploi</option>
            </select>
          </div>
          <div className="form-group">
            <label>Contrat</label>
            <select
              className="input"
              value={contract}
              onChange={(e) => {
                markEdit();
                setContract(e.target.value);
              }}
            >
              <option value="permanent">CDI</option>
              <option value="temporary">CDD</option>
              <option value="none">Aucun</option>
            </select>
          </div>
        </div>
        <div className="grid two">
          <div className="form-group">
            <label>Ancienneté (années)</label>
            <input
              className="input"
              type="number"
              value={seniority}
              onChange={(e) => {
                markEdit();
                setSeniority(Number(e.target.value));
              }}
            />
          </div>
          <div className="form-group">
            <label>Situation familiale</label>
            <select
              className="input"
              value={family}
              onChange={(e) => {
                markEdit();
                setFamily(e.target.value);
              }}
            >
              <option value="single">Célibataire</option>
              <option value="married">Marié</option>
            </select>
          </div>
        </div>
        <div className="grid two">
          <div className="form-group">
            <label>Conjoint employé</label>
            <select
              className="input"
              value={spouseEmployed}
              onChange={(e) => {
                markEdit();
                setSpouseEmployed(e.target.value as "unknown" | "yes" | "no");
              }}
            >
              <option value="unknown">Non renseigné</option>
              <option value="yes">Oui</option>
              <option value="no">Non</option>
            </select>
          </div>
          <div className="form-group">
            <label>Logement</label>
            <select
              className="input"
              value={housingStatus}
              onChange={(e) => {
                markEdit();
                setHousingStatus(e.target.value);
              }}
            >
              <option value="owner">Propriétaire</option>
              <option value="rent">Locataire</option>
              <option value="family">Famille</option>
            </select>
          </div>
        </div>
        <div className="form-group">
          <label>Titulaire principal</label>
          <select
            className="input"
            value={isPrimaryHolder ? "yes" : "no"}
            onChange={(e) => {
              markEdit();
              setIsPrimaryHolder(e.target.value === "yes");
            }}
          >
            <option value="yes">Oui</option>
            <option value="no">Non</option>
          </select>
        </div>
        <div className="form-group">
          <label>Documents (upload)</label>
          <input
            className="input"
            type="file"
            multiple
          onChange={(e) => {
            const files = Array.from(e.target.files || []);
            if (documents.length > 0 && files.length > 0) {
              docReuploadRef.current += 1;
            }
            setDocuments(files);
            if (files.length > 0) {
              setDocumentNames(files.map((f) => f.name).join(", "));
            }
          }}
        />
        {documents.length === 0 && (
          <input
            className="input"
            style={{ marginTop: 8 }}
            placeholder="Ou liste des documents (ex: salary.pdf, contract.pdf)"
            value={documentNames}
            onChange={(e) => {
              markEdit();
              setDocumentNames(e.target.value);
            }}
          />
        )}
      </div>
        {error && <div style={{ color: "#b91c1c", fontSize: 14 }}>{error}</div>}
        <button className="button-primary" type="submit" disabled={loading}>
          {loading ? "Envoi..." : "Soumettre"}
        </button>
      </form>
    </div>
  );
};
