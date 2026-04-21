# Phase M Benchmark Role Note

Date: 2026-04-20

This note locks benchmark roles for the clean learning protocol.

1. Benchmark-derived data used in prior stages (L1-L2.5) is development-only and does not define the paper-grade train/validation protocol.
2. Benchmark 61-90 is the clean primary external benchmark family for final test reporting in this branch.
3. Benchmark 1-60 is secondary robustness-only evaluation, treated as out-of-distribution or legacy transfer.
4. Benchmark instances are excluded from synthetic training and synthetic validation.

Operational rule:

- train/val manifests must reference generated synthetic instances only.
- benchmark manifests are test-only and split by role (`primary_vls`, `secondary_legacy`).
