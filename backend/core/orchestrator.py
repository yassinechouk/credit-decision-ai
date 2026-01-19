from agents.document_agent import analyze_documents
from agents.similarity_agent import analyze_similarity
from agents.decision_agent import make_decision
from agents.explanation_agent import explain_decision


def run_orchestrator(request):
    doc_result = analyze_documents(request)
    sim_result = analyze_similarity(request)

    decision = make_decision(doc_result, sim_result)
    explanation = explain_decision(decision, doc_result, sim_result)

    return {
        "decision": decision,
        "explanation": explanation,
    }
