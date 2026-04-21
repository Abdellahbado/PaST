# Results

Main diagnostic runs executed at `61/347` with variant `phaseI_noscreen_diagnostic`.

Bounded no-screen exact-eval cap ladder:

- cap `64`: start `6944`, best `6922`, improving move found = yes (`insert_inter`)
- cap `256`: start `6944`, best `6920`, improving move found = yes (`insert_inter`)
- cap `1024`: start `6944`, best `6920`, improving move found = yes (`insert_inter`)

Evaluated moves (exact, no shortlist pruning in tested batch):

- cap `64`: insert `48`, swap `16` (total `64`)
- cap `256`: insert `149`, swap `0` (total `149`) due to early improving move acceptance
- cap `1024`: insert `359`, swap `0` (total `359`) due to early improving move acceptance

Interpretation:

- evidence supports "screening too aggressive" for Phase H implementation at this point, because improving 1-move(s) appear under no-screen exact evaluation from the same start.
- current result does not support "true 1-move local optimum" under tested neighborhoods.
