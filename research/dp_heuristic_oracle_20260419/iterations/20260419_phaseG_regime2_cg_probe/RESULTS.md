# Results

Main run executed:

- instance `61`, epsilon `347`

Key outputs:

- initial columns: `259`
- iterations: `2`
- final columns: `260`
- LP bound: `7024.761959`
- restricted master IP TEC: `7040`
- paper EHS gap: `+330`
- reference/F2-init gap: `+397`

Baselines at same epsilon:

- `greedy_dp`: `7088`
- `greedy_dp_local_search_relocate_only`: `7081`

Follow-up corrected rerun:

- duplicate-stop logic fixed to search for best non-duplicate improving column
- pricing iterations: `12`
- final columns: `271` (added `12` genuinely new columns after the initial pool)
- LP bound: `7024.761959` (unchanged)
- restricted master IP TEC: `7040` (unchanged)
- stop reason: `max_iter`

Interpretation:

- corrected loop no longer stalls on duplicate and can continue adding valid columns.
- despite this, quality did not move on `61/347`; signal remains non-competitive versus EHS/reference in current bounded setup.
