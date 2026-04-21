# Problem

Phase D question:

- can we recover part of paper EHS strength by adding history-aware repair across decreasing `epsilon` (paper `Khat`), while keeping DP in machine-local roles?

Hypothesis:

- DP is not strong as global assignment constructor from scratch.
- DP is strong as:
  - machine-level ranking signal (safe LB / quick score)
  - exact touched-machine evaluator
  - post-repair local-improvement oracle

Target method family:

- seed at `epsilon_start` with one-shot `greedy_dp_local_search_relocate_only`
- step `epsilon: e+1 -> e` by repair:
  - remove assignments violating new `epsilon`
  - reinsert displaced jobs using DP-guided candidate ranking and exact touched-machine checks
  - optional relocate-only cleanup

Evaluation requirement:

- align transitions to paper EHS stored frontier points for instances `46`, `61`, `64`, `90`
- compare against same-`epsilon` paper EHS (when present), plus one-shot `greedy_dp` and one-shot `greedy_dp_local_search_relocate_only`
