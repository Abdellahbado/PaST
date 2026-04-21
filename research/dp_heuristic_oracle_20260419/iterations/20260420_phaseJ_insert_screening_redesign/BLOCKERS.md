# Blockers

Observed blockers:

- `vnd_exact_dp_insert_rank_diverse` has high memory pressure (~1.06 GB RSS), indicating pool-size/rerank overhead must be bounded tighter.
- evaluated insert counts remain high (`29160`), so ranking quality improved enough for acceptance but screening efficiency is still weak.
- `swap_inter` still offers no useful signal in this branch and remains secondary.

Implication:

- continue only with bounded efficiency tuning on the successful insert-ranking line, not broad method changes.
