# Problem

Phase Q established that the current supervised move-ranking formulation is not strong enough to justify continued investment as the main paper direction.

The next question is algorithmic:

- does warm-starting VND across nearby epsilon values recover cumulative improvements that fresh-start VND misses?

This branch is a bounded diagnostic, not a full new heuristic paper implementation.

The experiment must isolate one mechanism:

- same VND family,
- same exact DP local oracle,
- same instance family entry point,
- warm-started epsilon chain versus fresh-start control.

If warm-start consistently improves TEC on the same epsilon ladder, the next branch should scale that idea. If not, the warm-start hypothesis is weakened and another bottleneck must be targeted.
