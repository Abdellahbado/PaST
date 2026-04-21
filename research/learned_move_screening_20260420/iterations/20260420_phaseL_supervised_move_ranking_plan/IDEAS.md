# Ideas

## Primary idea

- supervised ranking of `insert_inter` moves using cheap analytical features and exact DP labels

## Secondary ideas kept in reserve

- binary classifier only as a later comparison, not as the first online design
- tiny tabular MLP as a secondary baseline to boosted trees
- analytical fallback + learned top-k hybrid shortlist

## Not-first ideas

- RL
- transformers
- GNNs
- ADP / value function approximation
- learned full assignment construction
