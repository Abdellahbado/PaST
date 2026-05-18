# Problem

Hard irregular `K=12` currently has an unacceptable failure mode: some rows return no feasible incumbent (`UB=-1`) after spending the full budget.

This is not acceptable for the paper or for practical scalability. A large but finite gap is acceptable; no schedule is not.

The evidence shows that `K=12` is not hopeless:

- some `hardA_k12` rows already produce finite gaps around `0.02%–0.05%`;
- some `hardB_k12` rows also produce finite gaps;
- other `hardB_k12` seeds timeout with no incumbent.

Therefore the next target is not exact closure. The target is an anytime-safe feasible incumbent.

## Core hypothesis

The current pipeline can spend the budget inside hard-K profile/beam/exact paths before preserving a feasible UB. We can fix this by computing a cheap valid incumbent before long Step-3/Step-4 work and returning it if later stages fail.

## Study hypothesis

At fixed `K=12`, difficulty depends strongly on arithmetic structure. We need a K=12 family panel, not only one hard size set.

