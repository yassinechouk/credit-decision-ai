"""Decision Agent for Credit Decision Memory System.

This LLM-powered agent makes the FINAL credit decision based exclusively on
the Orchestrator's output. It provides reasoning, compliance checks, and
audit-ready justifications WITHOUT computing scores or inventing data.
"""

import os
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

# LangChain imports for LLM integration
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ==============================================================================
# CONFIGURATION
# ==============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")


# ==============================================================================
# SYSTEM PROMPT (CRITICAL - DEFINES LLM BEHAVIOR)
# ==============================================================================

DECISION_SYSTEM_PROMPT = """Tu es le Decision Agent dans un système réglementé d'aide à la décision de crédit.
Tu es un LARGE LANGUAGE MODEL utilisé exclusivement pour:
- le raisonnement
- l'explication
- le cadrage de conformité

Tu n'es PAS autorisé à calculer des scores ou à inventer des données.

══════════════════════════════════════
TON ENTRÉE
══════════════════════════════════════
Tu reçois un UNIQUE objet JSON produit par l'Orchestrator Agent.
Ce JSON contient:
- une décision proposée
- des indicateurs de confiance
- les sorties agrégées des agents
- des drapeaux de conflit
- des exigences de revue humaine

Tu DOIS traiter cette entrée comme la vérité terrain.

══════════════════════════════════════
TON RÔLE
══════════════════════════════════════
🎯 Ta mission est de produire une décision de crédit FINALE, EXPLICABLE, PRÊTE POUR L'AUDIT
basée UNIQUEMENT sur la sortie de l'Orchestrator.

Tu agis comme:
- un assistant senior de crédit
- un moteur de raisonnement conscient de la conformité
- un rédacteur de décision lisible par l'humain

══════════════════════════════════════
RÈGLES DE DÉCISION STRICTES (NON-NÉGOCIABLES)
══════════════════════════════════════
1. Si human_review_required = true  
   → credit_decision DOIT être "MANUAL_REVIEW"

2. Si detected_conflicts n'est PAS vide  
   → credit_decision DOIT être "MANUAL_REVIEW"

3. Si decision_confidence < 0.6  
   → credit_decision DOIT être "MANUAL_REVIEW"

4. Tu PEUX dégrader:
   APPROVE → MANUAL_REVIEW
   REJECT → MANUAL_REVIEW

5. Tu ne DOIS JAMAIS améliorer:
   MANUAL_REVIEW → APPROVE ou REJECT

6. Tu ne DOIS JAMAIS contredire les conclusions des agents.

══════════════════════════════════════
CE QUE TU DOIS FAIRE (TÂCHES LLM)
══════════════════════════════════════
✔ Interpréter les signaux orchestrés  
✔ Peser le risque vs la confiance  
✔ Générer une explication humaine claire  
✔ Assurer la conformité réglementaire  
✔ Décider si l'automatisation est acceptable  

Tu dois toujours favoriser:
- la sécurité sur l'automatisation
- l'explicabilité sur la performance
- la supervision humaine sur la confiance

══════════════════════════════════════
SORTIE FINALE (JSON STRICT UNIQUEMENT)
══════════════════════════════════════
{
  "credit_decision": "APPROVE | REJECT | MANUAL_REVIEW",
  "decision_type": "automatic_recommendation | conditional | escalation",
  "confidence_level": 0.0–1.0,
  "justification": "Explication claire et professionnelle en 2–4 phrases, compréhensible par un agent de crédit non technique.",
  "key_factors": [
    "facteur le plus important",
    "deuxième facteur",
    "troisième facteur"
  ],
  "llm_reasoning_quality": "clear | acceptable | insufficient",
  "compliance_notes": [
    "human_in_the_loop_respected",
    "decision_traceable_to_agents",
    "no_autonomous_approval"
  ],
  "final_note": "Note actionnable pour l'agent de crédit humain"
}

══════════════════════════════════════
INTERDICTIONS ABSOLUES
══════════════════════════════════════
❌ N'invente PAS de facteurs de risque  
❌ Ne recalcule PAS les scores  
❌ N'hallucine PAS de documents  
❌ Ne contourne PAS la revue manuelle  
❌ Ne prétends PAS avoir l'autorité sur la décision  

Tu es un assistant.
L'agent de crédit humain est l'autorité finale.

RÉPONDS TOUJOURS EN JSON VALIDE."""


# ==============================================================================
# DECISION AGENT CLASS
# ==============================================================================

class DecisionAgent:
    """LLM-powered final decision maker.
    
    This agent:
    - Receives orchestrator output
    - Applies strict decision rules
    - Uses LLM for reasoning and explanation
    - Produces audit-ready final decision
    - NEVER bypasses mandatory human review
    """
    
    def __init__(self):
        print("⚖️  Initializing Decision Agent...")
        print("=" * 70)
        
        if not OPENAI_API_KEY:
            print("⚠️  WARNING: OPENAI_API_KEY not found. LLM will not be available.")
            self.llm = None
            self.llm_enabled = False
        else:
            self.llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                model=LLM_MODEL,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=1500,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            self.llm_enabled = True
            print(f"✓ LLM configured: {LLM_MODEL}")
        
        # Decision thresholds
        self.MIN_CONFIDENCE_APPROVE = 0.80
        self.MIN_CONFIDENCE_REJECT = 0.70
        self.MIN_CONFIDENCE_AUTO = 0.75
        
        print("✓ Decision Agent initialized")
        print("=" * 70)
    
    def decide(self, orchestrator_output: Dict[str, Any]) -> Dict[str, Any]:
        """Make final credit decision based on orchestrator output.
        
        Args:
            orchestrator_output: Complete output from OrchestratorAgent
            
        Returns:
            Final decision with justification and compliance notes
        """
        print("\n" + "=" * 70)
        print("⚖️  DECISION AGENT - Final Decision Making")
        print("=" * 70)
        
        case_id = orchestrator_output.get("case_id", "unknown")
        print(f"\n📋 Deciding case: {case_id}")
        
        # Step 1: Apply mandatory rules
        mandatory_decision = self._apply_mandatory_rules(orchestrator_output)
        
        if mandatory_decision:
            print(f"\n🔒 Mandatory rule triggered: {mandatory_decision['credit_decision']}")
            return mandatory_decision
        
        # Step 2: Use LLM for reasoning (if available)
        if self.llm_enabled:
            print("\n🤖 Invoking LLM for decision reasoning...")
            llm_decision = self._llm_decide(orchestrator_output)
            
            # Step 3: Validate LLM output against rules
            validated_decision = self._validate_llm_decision(llm_decision, orchestrator_output)
            return validated_decision
        else:
            print("\n⚠️  LLM not available, using rule-based fallback")
            return self._fallback_decision(orchestrator_output)
    
    def _apply_mandatory_rules(self, orch_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply non-negotiable decision rules.
        
        Returns decision dict if mandatory rule applies, None otherwise.
        """
        human_review_required = orch_output.get("human_review_required", False)
        conflicts = orch_output.get("detected_conflicts", [])
        confidence = orch_output.get("decision_confidence", 0.0)
        
        # Rule 1: Human review explicitly required
        if human_review_required:
            return self._create_decision(
                decision="MANUAL_REVIEW",
                decision_type="escalation",
                confidence=confidence,
                justification="Revue humaine obligatoire détectée par l'orchestrateur en raison de conditions critiques.",
                key_factors=["human_review_required", "compliance_requirement"],
                reasoning_quality="clear",
                compliance_notes=[
                    "human_in_the_loop_respected",
                    "mandatory_review_triggered",
                    "no_autonomous_bypass"
                ],
                final_note="Ce dossier nécessite l'examen d'un agent de crédit avant toute décision."
            )
        
        # Rule 2: Conflicts detected
        if conflicts:
            conflict_descriptions = [c.get("type", "unknown") for c in conflicts]
            return self._create_decision(
                decision="MANUAL_REVIEW",
                decision_type="escalation",
                confidence=confidence,
                justification=f"Conflits détectés entre agents: {', '.join(conflict_descriptions[:2])}. Arbitrage humain requis.",
                key_factors=["agent_conflicts", "inconsistent_signals"],
                reasoning_quality="clear",
                compliance_notes=[
                    "conflict_resolution_required",
                    "human_arbitration_needed",
                    "decision_traceable_to_agents"
                ],
                final_note=f"Résoudre les {len(conflicts)} conflit(s) identifié(s) avant décision."
            )
        
        # Rule 3: Low confidence
        if confidence < 0.6:
            return self._create_decision(
                decision="MANUAL_REVIEW",
                decision_type="escalation",
                confidence=confidence,
                justification=f"Confiance globale insuffisante ({confidence:.2%}). Revue manuelle nécessaire pour garantir la qualité de la décision.",
                key_factors=["low_confidence", "insufficient_data_quality"],
                reasoning_quality="acceptable",
                compliance_notes=[
                    "confidence_threshold_not_met",
                    "human_review_mandatory",
                    "quality_assurance_required"
                ],
                final_note="Améliorer la qualité des données ou procéder à une évaluation manuelle complète."
            )
        
        return None  # No mandatory rule applies
    
    def _llm_decide(self, orch_output: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to reason about the decision."""
        
        # Build prompt for LLM
        prompt = self._build_llm_prompt(orch_output)
        
        try:
            messages = [
                SystemMessage(content=DECISION_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            llm_decision = json.loads(response.content)
            
            print("✓ LLM reasoning completed")
            return llm_decision
            
        except Exception as e:
            print(f"❌ LLM invocation failed: {e}")
            return self._fallback_decision(orch_output)
    
    def _build_llm_prompt(self, orch_output: Dict[str, Any]) -> str:
        """Build structured prompt for LLM decision making."""
        
        case_id = orch_output.get("case_id", "unknown")
        proposed = orch_output.get("proposed_decision", "MANUAL_REVIEW")
        confidence = orch_output.get("decision_confidence", 0.0)
        human_review = orch_output.get("human_review_required", False)
        conflicts = orch_output.get("detected_conflicts", [])
        risk_indicators = orch_output.get("risk_indicators", [])
        aggregated = orch_output.get("aggregated_signals", {})
        
        prompt = f"""
# ORCHESTRATOR OUTPUT ANALYSIS

## Case Information
- Case ID: {case_id}
- Proposed Decision: {proposed}
- Global Confidence: {confidence:.2%}
- Human Review Required: {human_review}

## Conflicts Detected ({len(conflicts)})
{self._format_conflicts(conflicts)}

## Risk Indicators ({len(risk_indicators)})
{', '.join(risk_indicators[:10]) if risk_indicators else 'None'}

## Aggregated Signals
- Document Flags: {', '.join(aggregated.get('document_flags', [])[:5])}
- Behavior Flags: {', '.join(aggregated.get('behavior_flags', [])[:5])}
- Fraud Flags: {', '.join(aggregated.get('fraud_flags', [])[:5])}

## Risk Levels
{self._format_risk_levels(aggregated.get('risk_levels', {}))}

## Scores
{self._format_scores(aggregated.get('scores', {}))}

---

# YOUR TASK
Based EXCLUSIVELY on the above orchestrator output, produce a final credit decision.

Remember:
- If human_review_required = true → MUST be MANUAL_REVIEW
- If conflicts exist → MUST be MANUAL_REVIEW  
- If confidence < 0.6 → MUST be MANUAL_REVIEW
- You can downgrade but NEVER upgrade
- Provide clear justification in French
- List 3 key factors maximum
- Include actionable note for human officer

Respond ONLY with valid JSON following the schema in your system prompt.
"""
        return prompt
    
    def _format_conflicts(self, conflicts: List[Dict[str, Any]]) -> str:
        if not conflicts:
            return "None"
        lines = []
        for c in conflicts[:5]:
            lines.append(f"  - {c.get('type')} (severity: {c.get('severity')}): {c.get('description')}")
        return "\n".join(lines)
    
    def _format_risk_levels(self, risk_levels: Dict[str, str]) -> str:
        if not risk_levels:
            return "None"
        return "\n".join([f"  - {k}: {v}" for k, v in risk_levels.items()])
    
    def _format_scores(self, scores: Dict[str, float]) -> str:
        if not scores:
            return "None"
        return "\n".join([f"  - {k}: {v:.2f}" for k, v in scores.items()])
    
    def _validate_llm_decision(
        self, llm_decision: Dict[str, Any], orch_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and enforce compliance of LLM decision."""
        
        credit_decision = llm_decision.get("credit_decision", "MANUAL_REVIEW")
        
        # Enforce mandatory rules (in case LLM tried to bypass)
        human_review = orch_output.get("human_review_required", False)
        conflicts = orch_output.get("detected_conflicts", [])
        confidence = orch_output.get("decision_confidence", 0.0)
        
        if human_review and credit_decision != "MANUAL_REVIEW":
            print("⚠️  LLM tried to bypass mandatory human review - enforcing MANUAL_REVIEW")
            llm_decision["credit_decision"] = "MANUAL_REVIEW"
            llm_decision["decision_type"] = "escalation"
            llm_decision["compliance_notes"].append("llm_override_blocked")
        
        if conflicts and credit_decision != "MANUAL_REVIEW":
            print("⚠️  LLM ignored conflicts - enforcing MANUAL_REVIEW")
            llm_decision["credit_decision"] = "MANUAL_REVIEW"
            llm_decision["decision_type"] = "escalation"
            llm_decision["compliance_notes"].append("conflict_override_blocked")
        
        if confidence < 0.6 and credit_decision != "MANUAL_REVIEW":
            print("⚠️  LLM ignored low confidence - enforcing MANUAL_REVIEW")
            llm_decision["credit_decision"] = "MANUAL_REVIEW"
            llm_decision["decision_type"] = "escalation"
            llm_decision["compliance_notes"].append("confidence_override_blocked")
        
        # Add metadata
        llm_decision["case_id"] = orch_output.get("case_id")
        llm_decision["decision_timestamp"] = datetime.now().isoformat()
        llm_decision["orchestrator_confidence"] = confidence
        
        return llm_decision
    
    def _fallback_decision(self, orch_output: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based fallback when LLM is unavailable."""
        
        proposed = orch_output.get("proposed_decision", "MANUAL_REVIEW")
        confidence = orch_output.get("decision_confidence", 0.0)
        risk_indicators = orch_output.get("risk_indicators", [])
        
        # Conservative approach: default to MANUAL_REVIEW
        return self._create_decision(
            decision="MANUAL_REVIEW",
            decision_type="escalation",
            confidence=confidence,
            justification="Système d'analyse AI indisponible. Revue manuelle obligatoire pour garantir la qualité de la décision.",
            key_factors=["llm_unavailable", "conservative_approach", "quality_assurance"],
            reasoning_quality="insufficient",
            compliance_notes=[
                "llm_unavailable",
                "human_review_mandatory",
                "fallback_mode_activated"
            ],
            final_note="Analyser manuellement le dossier - système AI non disponible."
        )
    
    def _create_decision(
        self,
        decision: str,
        decision_type: str,
        confidence: float,
        justification: str,
        key_factors: List[str],
        reasoning_quality: str,
        compliance_notes: List[str],
        final_note: str
    ) -> Dict[str, Any]:
        """Create standardized decision output."""
        
        return {
            "credit_decision": decision,
            "decision_type": decision_type,
            "confidence_level": round(confidence, 4),
            "justification": justification,
            "key_factors": key_factors[:3],  # Limit to 3
            "llm_reasoning_quality": reasoning_quality,
            "compliance_notes": compliance_notes,
            "final_note": final_note,
            "decision_timestamp": datetime.now().isoformat(),
            "agent_version": "decision_agent_v1.0"
        }


# ==============================================================================
# WRAPPER FUNCTIONS
# ==============================================================================

_decision_agent_instance = None

def get_decision_agent() -> DecisionAgent:
    global _decision_agent_instance
    if _decision_agent_instance is None:
        _decision_agent_instance = DecisionAgent()
    return _decision_agent_instance


def make_final_decision(orchestrator_output: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for final decision making."""
    return get_decision_agent().decide(orchestrator_output)


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TEST DECISION AGENT")
    print("=" * 70)
    
    # Mock orchestrator output with conflicts
    mock_orchestrator_output = {
        "case_id": "TEST_DECISION_001",
        "proposed_decision": "APPROVE",
        "decision_confidence": 0.55,  # Low confidence
        "human_review_required": False,
        "detected_conflicts": [
            {
                "type": "doc_similarity_conflict",
                "severity": "high",
                "description": "Document LOW consistency but Similarity APPROVE",
                "agents": ["document_agent", "similarity_agent"]
            }
        ],
        "aggregated_signals": {
            "document_flags": ["INCOME_MISMATCH"],
            "behavior_flags": ["MULTIPLE_EDITS"],
            "fraud_flags": [],
            "scores": {
                "document_consistency": 0.65,
                "behavior_risk": 0.45,
                "similarity_risk": 0.35
            },
            "risk_levels": {
                "document": "MEDIUM",
                "behavior": "MEDIUM",
                "similarity": "modere"
            }
        },
        "risk_indicators": ["INCOME_MISMATCH", "MULTIPLE_EDITS"],
        "review_triggers": ["conflicts_detected_1", "low_confidence_0.55"]
    }
    
    decision = make_final_decision(mock_orchestrator_output)
    
    print("\n" + "=" * 70)
    print("FINAL DECISION")
    print("=" * 70)
    print(json.dumps(decision, indent=2, ensure_ascii=False))