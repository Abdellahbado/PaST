# PLAN19 Phase A Diagnosis: where closure is lost after incumbent production

## Evidence from PLAN18 (n=1000, lambda=1.3, seeds 0-3)

### hardA_k10

**Baseline (energy_core):**
- seed=0: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4
- seed=1: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4
- seed=2: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4
- seed=3: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4

**Reroute (profile_repair_beam + auto_v1):**
- seed=0: finite gap 0.0172%, deciding_step=step4, rt=338.9235s
- seed=1: finite gap 0.0272%, deciding_step=step4, rt=650.7857s
- seed=2: finite gap 0.0199%, deciding_step=step4, rt=356.0239s
- seed=3: finite gap 0.0358%, deciding_step=step4, rt=606.2630s

Diagnosis: beam produces incumbents but Step 4 exact DP does not close. Gaps: 0.0172% - 0.0358%.

### hardA_k12

**Baseline (energy_core):**
- seed=0: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4
- seed=1: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4
- seed=2: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4
- seed=3: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4

**Reroute (profile_repair_beam + auto_v1):**
- seed=0: finite gap 0.0239%, deciding_step=step4, rt=758.4520s
- seed=1: timeout (no incumbent), rt=1200.0000s
- seed=2: finite gap 0.0399%, deciding_step=step4, rt=721.4208s
- seed=3: timeout (no incumbent), rt=1200.0000s

Diagnosis: beam produces incumbents but Step 4 exact DP does not close. Gaps: 0.0239% - 0.0399%.

### hardB_k10

**Baseline (energy_core):**
- seed=0: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4
- seed=1: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4
- seed=2: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4
- seed=3: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4

**Reroute (profile_repair_beam + auto_v1):**
- seed=0: finite gap 0.0391%, deciding_step=step4, rt=755.4463s
- seed=1: finite gap 0.0620%, deciding_step=step3, rt=1266.0402s
- seed=2: finite gap 0.0450%, deciding_step=step4, rt=807.3616s
- seed=3: timeout (no incumbent), rt=1200.0000s

Diagnosis: beam produces incumbents but Step 4 exact DP does not close. Gaps: 0.0391% - 0.0620%.

### hardB_k12

**Baseline (energy_core):**
- seed=0: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4
- seed=1: no incumbent, deciding_step=external_timeout
- seed=2: selector bypass (non_mainline_solver), no incumbent, deciding_step=step4
- seed=3: no incumbent, deciding_step=external_timeout

**Reroute (profile_repair_beam + auto_v1):**
- seed=0: timeout (no incumbent), rt=1200.0000s
- seed=2: finite gap 0.0292%, deciding_step=step3, rt=1227.7878s
- seed=1: timeout (no incumbent), rt=1200.0000s
- seed=3: timeout (no incumbent), rt=1200.0000s

Diagnosis: beam produces incumbents but Step 4 exact DP does not close. Gaps: 0.0292% - 0.0292%.

## Overall Conclusion

- K=10: baseline is always bypassed/no-incumbent. Reroute beam produces incumbents consistently, but Step 4 global exact DP fails to close. The bottleneck is **closure after incumbent production**.
- K=12: baseline is bypassed/no-incumbent. Reroute beam sometimes produces incumbents, sometimes times out. The bottleneck is a mix of **incumbent production** and **closure**.
- Therefore, the highest-value redesigns are: (1) bounded exact closure after beam incumbent on K=10; (2) routing override to skip useless baseline on K>=10; (3) optionally stronger beam for K=12 incumbent production.
