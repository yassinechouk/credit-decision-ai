"""Orchestrator Agent for Credit Decision Memory System.

This agent coordinates all specialized agents, aggregates their outputs,
detects conflicts, and prepares a unified input for the Decision Agent.
It does NOT make final credit decisions - only orchestrates the pipeline.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class AgentPriority(Enum):
    """Priority levels for agent outputs in case of conflicts."""
    CRITICAL = 1  # Document/Fraud agents
    HIGH = 2      # Similarity/Behavior agents
    MEDIUM = 3    # Secondary agents


@dataclass
class AgentOutput:
    """Standardized container for agent responses."""
    agent_name: str
    success: bool
    data: Dict[str, Any]
    confidence: float
    priority: AgentPriority
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


class OrchestratorAgent:
    """Coordinates multi-agent credit analysis pipeline.
    
    Responsibilities:
    - Execute agents in optimal order
    - Aggregate and normalize outputs
    - Detect conflicts and inconsistencies
    - Compute confidence metrics
    - Determine human review requirements
    - Prepare decision-ready payload
    """
    
    def __init__(self):
        print("🎼 Initializing Orchestrator Agent...")
        print("=" * 70)
        
        # Agent execution order (considering dependencies)
        self.execution_order = [
            "document_agent",
            "similarity_agent", 
            "behavior_agent",
            "fraud_agent",      # Optional if implemented
            "image_agent",      # Optional if implemented
        ]
        
        # Thresholds for decision quality
        self.MIN_CONFIDENCE_AUTO = 0.75
        self.MIN_CONFIDENCE_RECOMMEND = 0.60
        self.MAX_CONFLICT_SCORE = 0.30
        
        print("✓ Orchestrator initialized")
        print("=" * 70)
    
    def orchestrate(self, request: Dict[str, Any], agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Main orchestration logic.
        
        Args:
            request: Original credit request
            agent_results: Dict mapping agent_name -> agent response
            
        Returns:
            Unified orchestrator output for Decision Agent
        """
        print("\n" + "=" * 70)
        print("🎼 ORCHESTRATOR AGENT - Pipeline Execution")
        print("=" * 70)
        
        case_id = request.get("case_id", "unknown")
        print(f"\n📋 Processing case: {case_id}")
        
        # 1. Normalize agent outputs
        normalized_outputs = self._normalize_agent_outputs(agent_results)
        
        # 2. Detect conflicts
        conflicts = self._detect_conflicts(normalized_outputs)
        
        # 3. Aggregate signals
        aggregated_signals = self._aggregate_signals(normalized_outputs)
        
        # 4. Compute global confidence
        global_confidence = self._compute_global_confidence(normalized_outputs, conflicts)
        
        # 5. Determine risk indicators
        risk_indicators = self._extract_risk_indicators(normalized_outputs)
        
        # 6. Compute proposed decision
        proposed_decision = self._propose_decision(aggregated_signals, risk_indicators, global_confidence)
        
        # 7. Determine if human review is required
        human_review_required = self._requires_human_review(
            global_confidence, conflicts, risk_indicators, normalized_outputs
        )
        
        # 8. Prepare final payload
        orchestrator_output = {
            "case_id": case_id,
            "orchestration_metadata": {
                "agents_executed": list(agent_results.keys()),
                "agents_succeeded": [k for k, v in normalized_outputs.items() if v.success],
                "agents_failed": [k for k, v in normalized_outputs.items() if not v.success],
                "execution_timestamp": self._get_timestamp(),
            },
            "proposed_decision": proposed_decision,
            "decision_confidence": round(global_confidence, 4),
            "human_review_required": human_review_required,
            "detected_conflicts": conflicts,
            "aggregated_signals": aggregated_signals,
            "risk_indicators": risk_indicators,
            "agent_outputs": {
                name: {
                    "success": output.success,
                    "confidence": output.confidence,
                    "data": output.data,
                    "error": output.error
                }
                for name, output in normalized_outputs.items()
            },
            "review_triggers": self._identify_review_triggers(
                conflicts, risk_indicators, global_confidence, normalized_outputs
            )
        }
        
        self._log_orchestration_summary(orchestrator_output)
        
        return orchestrator_output
    
    def _normalize_agent_outputs(self, agent_results: Dict[str, Any]) -> Dict[str, AgentOutput]:
        """Convert raw agent responses to standardized AgentOutput objects."""
        normalized = {}
        
        for agent_name, result in agent_results.items():
            if not isinstance(result, dict):
                normalized[agent_name] = AgentOutput(
                    agent_name=agent_name,
                    success=False,
                    data={},
                    confidence=0.0,
                    priority=AgentPriority.MEDIUM,
                    error="Invalid result format"
                )
                continue
            
            # Extract confidence from various possible locations
            confidence = result.get("confidence", 0.0)
            if confidence == 0.0:
                confidence = result.get("decision_confidence", 0.0)
            if confidence == 0.0:
                confidence = result.get("explanation_confidence", 0.0)
            
            # Determine priority
            priority = self._get_agent_priority(agent_name)
            
            normalized[agent_name] = AgentOutput(
                agent_name=agent_name,
                success=True,
                data=result,
                confidence=float(confidence),
                priority=priority,
                error=None
            )
        
        print(f"\n✓ Normalized {len(normalized)} agent outputs")
        return normalized
    
    def _get_agent_priority(self, agent_name: str) -> AgentPriority:
        """Assign priority level based on agent type."""
        critical_agents = ["document_agent", "fraud_agent", "image_agent"]
        high_priority = ["similarity_agent", "behavior_agent"]
        
        if agent_name in critical_agents:
            return AgentPriority.CRITICAL
        elif agent_name in high_priority:
            return AgentPriority.HIGH
        return AgentPriority.MEDIUM
    
    def _detect_conflicts(self, outputs: Dict[str, AgentOutput]) -> List[Dict[str, Any]]:
        """Detect logical conflicts between agent conclusions."""
        conflicts = []
        
        # Example: Document agent flags HIGH inconsistency but Similarity suggests APPROVE
        doc_output = outputs.get("document_agent")
        sim_output = outputs.get("similarity_agent")
        
        if doc_output and sim_output:
            doc_data = doc_output.data.get("document_analysis", {})
            sim_data = sim_output.data.get("ai_analysis", {})
            
            doc_consistency = doc_data.get("consistency_level", "UNKNOWN")
            sim_recommendation = sim_data.get("recommendation", "REVISER")
            
            if doc_consistency == "LOW" and sim_recommendation in {"APPROUVER", "APPROVE"}:
                conflicts.append({
                    "type": "doc_similarity_conflict",
                    "severity": "high",
                    "description": "Document agent reports LOW consistency but Similarity agent recommends APPROVE",
                    "agents": ["document_agent", "similarity_agent"]
                })
        
        # Check behavior vs similarity
        behavior_output = outputs.get("behavior_agent")
        if behavior_output and sim_output:
            behavior_data = behavior_output.data.get("behavior_analysis", {})
            behavior_level = behavior_data.get("behavior_level", "UNKNOWN")
            sim_recommendation = sim_output.data.get("ai_analysis", {}).get("recommendation", "REVISER")
            
            if behavior_level == "HIGH" and sim_recommendation in {"APPROUVER", "APPROVE"}:
                conflicts.append({
                    "type": "behavior_similarity_conflict",
                    "severity": "medium",
                    "description": "Behavior agent reports HIGH risk but Similarity recommends APPROVE",
                    "agents": ["behavior_agent", "similarity_agent"]
                })
        
        # Check fraud flags
        fraud_output = outputs.get("fraud_agent")
        if fraud_output and fraud_output.success:
            fraud_flags = (
                fraud_output.data.get("fraud_analysis", {}).get("detected_flags")
                or fraud_output.data.get("fraud_flags", [])
            )
            if fraud_flags and sim_output:
                sim_recommendation = sim_output.data.get("ai_analysis", {}).get("recommendation", "REVISER")
                if sim_recommendation in {"APPROUVER", "APPROVE"}:
                    conflicts.append({
                        "type": "fraud_detected",
                        "severity": "critical",
                        "description": "Fraud signals detected despite positive recommendation",
                        "agents": ["fraud_agent", "similarity_agent"]
                    })
        
        if conflicts:
            print(f"\n⚠️  Detected {len(conflicts)} conflicts:")
            for conflict in conflicts:
                print(f"   - {conflict['type']} (severity: {conflict['severity']})")
        
        return conflicts
    
    def _aggregate_signals(self, outputs: Dict[str, AgentOutput]) -> Dict[str, Any]:
        """Aggregate key signals from all agents."""
        aggregated = {
            "document_flags": [],
            "behavior_flags": [],
            "fraud_flags": [],
            "image_flags": [],
            "similarity_patterns": [],
            "risk_levels": {},
            "scores": {}
        }
        
        # Document agent
        doc_output = outputs.get("document_agent")
        if doc_output and doc_output.success:
            doc_data = doc_output.data.get("document_analysis", {})
            aggregated["document_flags"] = doc_data.get("flags", [])
            aggregated["scores"]["document_consistency"] = doc_data.get("dds_score", 0.0)
            aggregated["risk_levels"]["document"] = doc_data.get("consistency_level", "UNKNOWN")
        
        # Behavior agent
        behavior_output = outputs.get("behavior_agent")
        if behavior_output and behavior_output.success:
            behavior_data = behavior_output.data.get("behavior_analysis", {})
            aggregated["behavior_flags"] = behavior_data.get("behavior_flags", [])
            aggregated["scores"]["behavior_risk"] = behavior_data.get("brs_score", 0.0)
            aggregated["risk_levels"]["behavior"] = behavior_data.get("behavior_level", "UNKNOWN")
        
        # Similarity agent
        sim_output = outputs.get("similarity_agent")
        if sim_output and sim_output.success:
            sim_data = sim_output.data.get("ai_analysis", {})
            rag_stats = sim_output.data.get("rag_statistics", {})
            aggregated["similarity_patterns"] = [sim_data.get("recommendation", "REVISER")]
            try:
                aggregated["scores"]["similarity_risk"] = float(sim_data.get("risk_score", 0.5))
            except (TypeError, ValueError):
                aggregated["scores"]["similarity_risk"] = 0.5
            aggregated["scores"]["peer_success_rate"] = rag_stats.get("repayment_success_rate", 0.0)
            aggregated["risk_levels"]["similarity"] = sim_data.get("risk_level", "modere")
        
        # Fraud agent (if present)
        fraud_output = outputs.get("fraud_agent")
        if fraud_output and fraud_output.success:
            fraud_analysis = fraud_output.data.get("fraud_analysis", {})
            aggregated["fraud_flags"] = fraud_analysis.get("detected_flags", []) or fraud_output.data.get("fraud_flags", [])
            aggregated["scores"]["fraud_risk"] = fraud_analysis.get("fraud_score", fraud_output.data.get("fraud_score", 0.0))

        # Image agent (if present)
        image_output = outputs.get("image_agent")
        if image_output and image_output.success:
            image_analysis = image_output.data.get("image_analysis", {})
            aggregated["image_flags"] = image_analysis.get("flags", [])
            aggregated["scores"]["image_risk"] = image_analysis.get("ifs_score", 0.0)
            aggregated["risk_levels"]["image"] = image_analysis.get("risk_level", "UNKNOWN")

        return aggregated
    
    def _compute_global_confidence(
        self, outputs: Dict[str, AgentOutput], conflicts: List[Dict[str, Any]]
    ) -> float:
        """Compute weighted global confidence across all agents."""
        
        if not outputs:
            return 0.0
        
        # Collect confidences with weights
        weighted_confidences = []
        
        for agent_name, output in outputs.items():
            if not output.success:
                continue
            
            # Weight by priority
            weight = 1.0
            if output.priority == AgentPriority.CRITICAL:
                weight = 2.0
            elif output.priority == AgentPriority.HIGH:
                weight = 1.5
            
            weighted_confidences.append((output.confidence, weight))
        
        if not weighted_confidences:
            return 0.0
        
        # Weighted average
        total_weight = sum(w for _, w in weighted_confidences)
        weighted_sum = sum(c * w for c, w in weighted_confidences)
        base_confidence = weighted_sum / total_weight
        
        # Penalty for conflicts
        conflict_penalty = 0.0
        for conflict in conflicts:
            if conflict.get("severity") == "critical":
                conflict_penalty += 0.3
            elif conflict.get("severity") == "high":
                conflict_penalty += 0.2
            elif conflict.get("severity") == "medium":
                conflict_penalty += 0.1
        
        # Penalty for failed agents
        failed_count = sum(1 for o in outputs.values() if not o.success)
        failure_penalty = failed_count * 0.1
        
        final_confidence = base_confidence - conflict_penalty - failure_penalty
        return max(0.0, min(1.0, final_confidence))
    
    def _extract_risk_indicators(self, outputs: Dict[str, AgentOutput]) -> List[str]:
        """Extract all risk indicators from agent outputs."""
        indicators = []
        
        for agent_name, output in outputs.items():
            if not output.success:
                continue
            
            # Document flags
            doc_data = output.data.get("document_analysis", {})
            indicators.extend(doc_data.get("flags", []))
            
            # Behavior flags
            behavior_data = output.data.get("behavior_analysis", {})
            indicators.extend(behavior_data.get("behavior_flags", []))
            
            # Fraud flags
            fraud_flags = output.data.get("fraud_analysis", {}).get("detected_flags")
            indicators.extend(fraud_flags or output.data.get("fraud_flags", []))

            # Image flags
            image_flags = output.data.get("image_analysis", {}).get("flags") or output.data.get("image_flags")
            if image_flags:
                indicators.extend(image_flags)
            
            # Similarity red flags
            sim_data = output.data.get("ai_analysis", {})
            indicators.extend(sim_data.get("red_flags", []))
        
        # Deduplicate
        return list(dict.fromkeys(indicators))
    
    def _propose_decision(
        self, signals: Dict[str, Any], risk_indicators: List[str], confidence: float
    ) -> str:
        """Propose a preliminary decision based on aggregated signals.
        
        Note: This is NOT the final decision - Decision Agent makes that call.
        This is just a recommendation based on rule-based logic.
        """
        
        # Critical blocking conditions
        fraud_flags = signals.get("fraud_flags", [])
        if fraud_flags:
            return "REJECT"
        
        doc_flags = signals.get("document_flags", [])
        critical_doc_flags = ["MISSING_DOCUMENTS", "INCOME_MISMATCH"]
        if any(flag in doc_flags for flag in critical_doc_flags):
            return "MANUAL_REVIEW"
        
        # Check similarity recommendation
        sim_recommendation = None
        if signals.get("similarity_patterns"):
            sim_recommendation = signals["similarity_patterns"][0]
        
        # Check risk scores
        doc_score = signals.get("scores", {}).get("document_consistency", 0.0)
        behavior_score = signals.get("scores", {}).get("behavior_risk", 0.0)
        similarity_risk = signals.get("scores", {}).get("similarity_risk", 0.5)
        
        # Decision logic
        if doc_score < 0.6 or behavior_score > 0.7 or similarity_risk > 0.7:
            return "REJECT"
        
        if sim_recommendation in {"APPROUVER", "APPROVE"} and confidence >= self.MIN_CONFIDENCE_AUTO:
            if doc_score >= 0.8 and behavior_score < 0.4:
                return "APPROVE"
        
        if sim_recommendation == "APPROUVER_AVEC_CONDITIONS":
            return "MANUAL_REVIEW"
        
        if sim_recommendation == "REFUSER":
            return "REJECT"
        
        # Default to manual review
        return "MANUAL_REVIEW"
    
    def _requires_human_review(
        self,
        confidence: float,
        conflicts: List[Dict[str, Any]],
        risk_indicators: List[str],
        outputs: Dict[str, AgentOutput]
    ) -> bool:
        """Determine if human review is mandatory."""
        
        # Low confidence always requires review
        if confidence < self.MIN_CONFIDENCE_RECOMMEND:
            return True
        
        # Any conflicts require review
        if conflicts:
            return True
        
        # Critical risk indicators
        critical_indicators = [
            "FRAUD", "INCOME_MISMATCH", "MISSING_DOCUMENTS", 
            "DOC_INCONSISTENCY", "MULTIPLE_EDITS", "DOC_TAMPER"
        ]
        if any(indicator in risk_indicators for indicator in critical_indicators):
            return True
        
        # Failed critical agents
        for agent_name, output in outputs.items():
            if not output.success and output.priority == AgentPriority.CRITICAL:
                return True
        
        return False
    
    def _identify_review_triggers(
        self,
        conflicts: List[Dict[str, Any]],
        risk_indicators: List[str],
        confidence: float,
        outputs: Dict[str, AgentOutput]
    ) -> List[str]:
        """Identify specific reasons why human review might be needed."""
        triggers = []
        
        if confidence < self.MIN_CONFIDENCE_RECOMMEND:
            triggers.append(f"low_confidence_{confidence:.2f}")
        
        if conflicts:
            triggers.append(f"conflicts_detected_{len(conflicts)}")
        
        critical_risks = ["FRAUD", "INCOME_MISMATCH", "MISSING_DOCUMENTS", "DOC_TAMPER"]
        if any(risk in risk_indicators for risk in critical_risks):
            triggers.append("critical_risk_indicators")
        
        failed_agents = [name for name, out in outputs.items() if not out.success]
        if failed_agents:
            triggers.append(f"agent_failures_{len(failed_agents)}")
        
        return triggers
    
    def _log_orchestration_summary(self, output: Dict[str, Any]):
        """Print orchestration summary."""
        print("\n" + "=" * 70)
        print("📊 ORCHESTRATION SUMMARY")
        print("=" * 70)
        print(f"   Proposed Decision: {output['proposed_decision']}")
        print(f"   Global Confidence: {output['decision_confidence']:.2%}")
        print(f"   Human Review Required: {output['human_review_required']}")
        print(f"   Conflicts Detected: {len(output['detected_conflicts'])}")
        print(f"   Risk Indicators: {len(output['risk_indicators'])}")
        print(f"   Review Triggers: {', '.join(output['review_triggers']) if output['review_triggers'] else 'None'}")
        print("=" * 70)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


# ==============================================================================
# WRAPPER FUNCTIONS
# ==============================================================================

_orchestrator_instance = None

def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = OrchestratorAgent()
    return _orchestrator_instance


def orchestrate_decision(request: Dict[str, Any], agent_results: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for orchestration."""
    return get_orchestrator().orchestrate(request, agent_results)


# ==============================================================================
# TEST
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TEST ORCHESTRATOR AGENT")
    print("=" * 70)
    
    # Mock agent results
    test_request = {"case_id": "TEST_001"}
    
    mock_agent_results = {
        "document_agent": {
            "case_id": "TEST_001",
            "document_analysis": {
                "dds_score": 0.75,
                "consistency_level": "MEDIUM",
                "flags": ["INCOME_MISMATCH"],
                "missing_documents": []
            },
            "confidence": 0.7
        },
        "similarity_agent": {
            "case_id": "TEST_001",
            "ai_analysis": {
                "recommendation": "APPROUVER_AVEC_CONDITIONS",
                "risk_score": 0.4,
                "risk_level": "modere",
                "red_flags": []
            },
            "rag_statistics": {
                "repayment_success_rate": 0.82
            },
            "confidence": 0.75
        },
        "behavior_agent": {
            "case_id": "TEST_001",
            "behavior_analysis": {
                "brs_score": 0.35,
                "behavior_level": "MEDIUM",
                "behavior_flags": ["MULTIPLE_EDITS"]
            },
            "confidence": 0.65
        }
    }
    
    result = orchestrate_decision(test_request, mock_agent_results)
    
    print("\n" + "=" * 70)
    print("ORCHESTRATION RESULT")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
