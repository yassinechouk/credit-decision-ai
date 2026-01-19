def make_decision(doc_result, sim_result):
    if sim_result["fraud_ratio"] > 0.7:
        return "review"
    return "approve"
