# Blockers

Current status:

- no hard blocker for Phase C completion on the required rows

Active limitations:

- swap neighborhood can become large (`evaluated_swap_moves` high on some rows), increasing runtime without many accepted moves
- assignment-conditioned LB remains diagnostic only and should not be used as a global proof
- machine-priority signal quality is instance-dependent (helpful on `61/350`, harmful on `46/120`)

Mitigation direction:

- prioritize move-screening/pruning before exact DP swap evaluations if this branch is continued
- set `relocate_only` as default mode and keep priority-machines as optional targeted mode
