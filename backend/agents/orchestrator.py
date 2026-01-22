"""
PURE ORCHESTRATOR (No LLM)
Rôle : Chef d'orchestre logistique. 
Il collecte les données et prépare le dossier pour l'Agent de Décision.
"""

import operator
# Notez l'absence totale d'imports OpenAI ou LangChain LLM ici.
from typing import Dict, Any, List, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END

# ==============================================================================
# 1. ÉTAT DU WORKFLOW
# ==============================================================================

class OrchestratorState(TypedDict):
    case_id: str
    input_data: Dict[str, Any]
    
    # Stockage temporaire des résultats
    doc_output: Optional[Dict]
    beh_output: Optional[Dict]
    sim_output: Optional[Dict]
    
    # Sortie : Le Rapport (Pas de décision, juste des faits)
    final_report: Optional[Dict]
    
    logs: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]

# ==============================================================================
# 2. LOGIQUE TECHNIQUE (Pas d'IA, juste du code)
# ==============================================================================

def run_document_agent(state: OrchestratorState):
    """Appelle l'agent Document (OCR/Règles)."""
    # Ici, vous importeriez votre vrai module : from agents import doc_agent
    # C'est une tâche technique, pas besoin de LLM pour lancer un OCR.
    try:
        print("   [1/3] 📄 Scan des documents...")
        return {
            "doc_output": {"valid": True, "income_found": 4500}, 
            "logs": ["Doc Scan OK"]
        }
    except Exception as e:
        return {"errors": [f"Doc Error: {e}"]}

def run_behavioral_agent(state: OrchestratorState):
    """Appelle l'agent Comportemental (Analyse de logs)."""
    # Analyse des timestamps et clics. C'est des maths, pas de l'IA générative.
    try:
        print("   [2/3] 🖱️ Analyse comportementale...")
        return {
            "beh_output": {"risk_score": 0.2, "typing_speed": "Normal"},
            "logs": ["Behavior OK"]
        }
    except Exception as e:
        return {"errors": [f"Beh Error: {e}"]}

def run_similarity_agent(state: OrchestratorState):
    """Appelle l'agent RAG (Recherche vectorielle)."""
    # Interroge une base de données (ChromaDB/Pinecone). C'est de la recherche, pas de la génération.
    try:
        print("   [3/3] 🧠 Recherche d'historique (RAG)...")
        return {
            "sim_output": {"similar_cases": 10, "default_rate": 0.05},
            "logs": ["RAG OK"]
        }
    except Exception as e:
        return {"errors": [f"Sim Error: {e}"]}

# ==============================================================================
# 3. CONSOLIDATION (Packaging)
# ==============================================================================

def aggregator_node(state: OrchestratorState):
    """
    Rassemble tout dans un JSON propre.
    C'est ici que l'Orchestrateur passe le relais.
    """
    print("   [FIN] 📦 Création du rapport pour l'Agent Décision...")
    
    # On structure les données pour que le LLM (Agent Décision) n'ait plus qu'à lire
    report = {
        "metadata": {"case_id": state["case_id"]},
        "data": {
            "financial": state.get("doc_output"),
            "behavioral": state.get("beh_output"),
            "historical": state.get("sim_output")
        },
        "system_status": "ERROR" if state["errors"] else "HEALTHY"
    }
    
    return {"final_report": report}

# ==============================================================================
# 4. LE GRAPHE (Workflow)
# ==============================================================================

def build_pure_orchestrator():
    workflow = StateGraph(OrchestratorState)
    
    workflow.add_node("doc", run_document_agent)
    workflow.add_node("beh", run_behavioral_agent)
    workflow.add_node("sim", run_similarity_agent)
    workflow.add_node("aggregate", aggregator_node)
    
    # Démarre par Doc
    workflow.set_entry_point("doc")
    
    # Enchaînement simple
    workflow.add_edge("doc", "beh")
    workflow.add_edge("beh", "sim")
    workflow.add_edge("sim", "aggregate")
    
    workflow.add_edge("aggregate", END)
    
    return workflow.compile()

# ==============================================================================
# TEST RAPIDE
# ==============================================================================

if __name__ == "__main__":
    app = build_pure_orchestrator()
    result = app.invoke({"case_id": "123", "input_data": {}, "logs": [], "errors": []})
    
    import json
    print("\n✅ CE JSON SERA ENVOYÉ À L'AGENT DÉCISION (QUI CONTIENT LE LLM) :")
    print(json.dumps(result["final_report"], indent=2))