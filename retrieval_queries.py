# Centralized retrieval queries used by ContractRAG (NDA version)

GENERAL_SUMMARY_RETRIEVAL_QUERY = """
Summarize this NDA / confidentiality agreement.
Extract the most important information about:
the parties, effective date, purpose of disclosure,
definition of confidential information, exclusions,
obligations of the receiving party, permitted use,
term/duration, return or destruction of materials,
remedies, governing law, and risks.
"""

KEY_FIELDS_FALLBACK_RETRIEVAL_QUERY = """
Find the effective date, parties, purpose, definition of confidential information,
exclusions from confidentiality, confidentiality term, return or destruction clause,
governing law, and any non-solicitation or non-compete language.
"""

CONFIDENTIALITY_TERM_RETRIEVAL_QUERY = """
Find the confidentiality term, survival period, effective date,
termination date, duration of obligations, and any clause stating
how long confidentiality obligations continue after disclosure or termination.
"""

RISK_ANALYSIS_RETRIEVAL_QUERY = """
Find risky clauses in this NDA, especially about:
broad definition of confidential information,
one-sided obligations, long survival period,
unclear exclusions, injunctive relief, assignment,
non-solicitation, non-compete, indemnity,
governing law, and strict return or destruction obligations.
"""

RISK_QA_RETRIEVAL_QUERY = """
Find risky clauses, one-sided protections, broad confidentiality scope,
survival term, exclusions, return/destruction obligations,
injunctive relief, residuals, assignment, non-solicitation,
non-compete, penalties, and recipient responsibilities.
"""