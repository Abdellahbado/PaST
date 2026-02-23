"""G-LNS: Generative Large Neighborhood Search for bi-objective parallel machine
TOU scheduling.

Implements the G-LNS framework (Zhao et al., 2026) adapted for:
  - Bi-objective (Cmax, TEC) with Pareto archive
  - Parallel machine scheduling under Time-of-Use electricity pricing
  - LLM-generated destroy/repair operators via Groq (Kimi-K2)
  - DP timing layer for optimal start-time computation
"""

__version__ = "0.1.0"
