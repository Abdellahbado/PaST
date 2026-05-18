# Blockers

## Block 1: Coarsening degrades beam quality on hardA families

All coarsening variants are either same or strictly worse on hardA_k10. The beam performs best with the standard 14-block structure. Reducing to 8 or 12 blocks consistently increases the gap.

## Block 2: No single view generalizes across families

- `target_B8` improves hardB seeds 0-2 but worsens all hardA seeds
- `target_B12` improves hardA seed 2, hardB seeds 1-2 but worsens hardA seeds 1,3 and hardB seeds 0,3
- `price_preserve_B12` shows similar pattern to target_B12
- `arith_adaptive` is a no-op on hardA (threshold never triggers), mixed on hardB

The failure mode is consistent: coarsening removes too much price-profile structure. The beam's cost estimates become less accurate with wider blocks, and the lost precision outweighs any benefit from having fewer layers.

## Block 3: Coarser blocks increase pattern count, not decrease it

The original hypothesis was that fewer blocks → fewer beam decisions → better beam quality. In practice, wider blocks have larger capacities, which explode the number of candidate count patterns (`generate_energy_core_patterns`). The beam then has a larger search space per layer, not a smaller one.

## Resolution status

This direction is stopped (Decision C). No further work on adjacent block coarsening.
