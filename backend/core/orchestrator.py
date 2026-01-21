from agents.document_agent import analyze_documents
from agents.similarity_agent import analyze_similarity
from agents.behavior_agent import analyze_behavior
from agents.decision_agent import make_decision
from agents.explanation_agent import explain_decision


def run_orchestrator(request):
    doc_result = analyze_documents(request)
    behavior_result = analyze_behavior(request)
    sim_result = analyze_similarity(request)

    decision = make_decision(doc_result, sim_result, behavior_result)
    explanation = explain_decision(decision, doc_result, sim_result, behavior_result=behavior_result)

    return {
        "decision": decision,
        "explanation": explanation,
        "behavior": behavior_result,
    }
