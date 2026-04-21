# Ideas

## Core idea

Use the DP as a heuristic machine oracle, not as an exact global proof engine.

## Planned sequence

1. Implement a simple fixed-`epsilon` assignment baseline
2. Compare:
   - ESR per machine
   - exact DP per machine
3. Validate against exact fixed-`epsilon` values
4. Only if successful, test relaxed-DP-guided assignment

## What we deliberately avoid in this iteration

- full EHS replication
- exact global assignment search
- full `epsilon`-frontier implementation
