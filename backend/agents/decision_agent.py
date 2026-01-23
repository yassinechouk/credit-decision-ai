def make_decision(doc_result, sim_result, behavior_result=None):
    # Legacy simple rule preserved; behavior_result reserved for future use.
    if sim_result.get("fraud_ratio", 0) > 0.7:
        return "review"
    return "approve"
