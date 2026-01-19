# backend/agents

Chaque agent a un rôle clair et séparé.

- Document: décrit et extrait infos (montants, revenus, cohérence). Exemple: {"document_quality":"good","income_stability":"medium","extracted_monthly_income":2800,"flags":["revenu variable"]}
- Image: crédibilité visuelle (scan, signatures, mise en page). Exemple: {"visual_quality":"acceptable","document_consistency":"high","flags":["scan basse résolution"]}
- Comportement: sérieux et régularité. Exemple: {"behavior_profile":"normal","confidence_level":"high"}
- Similarité: comparaison aux cas passés (bons/mauvais). Exemple: {"similar_good_profiles":14,"similar_bad_profiles":3,"repayment_success_rate":0.82,"insight":"profil proche de freelances stables"}
- Fraude: indice de fraude documentaire. Exemple: {"fraud_risk":"low","fraud_probability":0.12}
- Décision: agrège les signaux et recommande. Exemple: {"credit_eligibility":"approved","recommended_amount":15000,"confidence":"high","main_reasons":["revenus stables","documents crédibles","profil similaire à clients solvables"]}
- Explication: formate les raisons de manière compréhensible.

Résultat attendu: des sorties structurées, neutres et explicables (pas d’accusation directe, pas de biais).
