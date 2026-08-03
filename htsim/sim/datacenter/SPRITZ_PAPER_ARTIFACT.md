# Spritz Artifact Reproduction Report

**Date:** 2026-02-11/12  
**Branch:** `ad`  
**Cluster:** CSCS Alps (clariden)  
**Account:** `a-g200`, partition `normal`

---

## Steps

### 1. Clone & Build
- Cloned `github.com/aleskubicek/sc25-spritz`, branch `ad`
- Built htsim library + datacenter executables (`htsim_uec_df`, `htsim_uec_sf`) with `make -j16`

### 2. Run Simulations via SLURM
- Submitted `reproduce.sh full` on 1 compute node (288 CPUs, 256GB RAM, 4h limit, `REPRO_PARALLEL=32`)
- All **28 experiments** completed (~2.5h): 14 DF (Dragonfly p4a8h4) + 14 SF (Slim Fly p7q9)
  - 7 no-fail: permutation 4MiB, adversarial 4MiB, adversarial 1MiB, allreduce ring, allreduce butterfly, alltoall, websearch
  - 7 fail-2%: same workloads with 2% random link failures

### 3. Generate Plots
- Created Python 3.11 venv (system Python 3.6 lacks `matplotlib.bar_label()`)
- Ran `reproduce.sh plot` for Fig 6, Fig 7, Fig 8
- Fixed missing subplots: added butterfly CCT, alltoall CCT (Fig 7), and all fail_2p experiments (Fig 8) to `reproduce.sh`
- Adversarial fail_2p required custom combined plot (DF uses `adv_i5_4MiB`, SF uses `adv_all_4MiB` — `plot.py` only accepts one experiment name)

### 4. Validation Re-run
- Fresh clone to separate directory (`sc25-spritz-validation`)
- Full pipeline from scratch: build + 28 sims + all plots
- No-fail experiments are **bit-identical** across both runs (0.0% diff)
- Fail experiments show expected stochastic variation (<5% for Spritz)

### 5. Commit & Push
- Committed updated `reproduce.sh` to branch `ad` (commit `43f71f5`)

---

## Generated Outputs

**Location:** `paper_plots/full/` — 18 PDFs + 18 PNGs (+1 extra DF-only adversarial)

### Figure 6 — No-fail microbenchmarks (4 subplots)
Stacked bar: FCT distribution + dropped packets + OOO%

| File | Subplot |
|------|---------|
| `fig6_df_permutation.pdf` | Dragonfly: Permutation (4MiB) |
| `fig6_df_adversarial.pdf` | Dragonfly: Adversarial (4MiB) |
| `fig6_sf_permutation.pdf` | Slim Fly: Permutation (4MiB) |
| `fig6_sf_adversarial.pdf` | Slim Fly: Adversarial (4MiB) |

### Figure 7 — AI collectives + DC traces (8 subplots)
Bar plots (CCT) + stacked bar (websearch FCT/dropped/OOO)

| File | Subplot |
|------|---------|
| `fig7_df_allreduce_bf_cct.pdf` | DF: Allreduce Butterfly CCT |
| `fig7_df_allreduce_ring_cct.pdf` | DF: Allreduce Ring CCT |
| `fig7_df_alltoall_cct.pdf` | DF: Alltoall CCT |
| `fig7_df_websearch.pdf` | DF: WebSearch (stacked bar) |
| `fig7_sf_allreduce_bf_cct.pdf` | SF: Allreduce Butterfly CCT |
| `fig7_sf_allreduce_ring_cct.pdf` | SF: Allreduce Ring CCT |
| `fig7_sf_alltoall_cct.pdf` | SF: Alltoall CCT |
| `fig7_sf_websearch.pdf` | SF: WebSearch (stacked bar) |

### Figure 8 — 2% link failures (6 subplots + 1 extra)
Violin plots of FCT distribution (only Valiant/OPS/Spritz — Minimal/UGAL-L/ECMP/Flicr fail to finish within 1s)

| File | Subplot |
|------|---------|
| `fig8_adversarial_fail_2p.pdf` | Adversarial DF+SF combined |
| `fig8_permutation_fail_2p.pdf` | Permutation DF+SF |
| `fig8_allreduce_ring_fail_2p.pdf` | Allreduce Ring DF+SF |
| `fig8_allreduce_bf_fail_2p.pdf` | Allreduce Butterfly DF+SF |
| `fig8_alltoall_fail_2p.pdf` | Alltoall DF+SF |
| `fig8_websearch_fail_2p.pdf` | WebSearch DF+SF |
| `fig8_adversarial_fail_2p_df.pdf` | *(extra)* Adversarial DF-only |

---

## Numerical Verification

### Fig 6 — Spritz speedup over next-best (avg FCT, no-fail)
| Experiment | Speedup | Paper claim |
|------------|---------|-------------|
| DF Permutation | ~1.0x (tied with UGAL-L) | 1.1-1.2x |
| DF Adversarial | 1.32x over Valiant | 1.3-1.5x |
| SF Permutation | Competitive (UGAL-L slightly better) | 1.1-1.2x |
| SF Adversarial | 1.14x over Valiant | 1.3-1.5x |

### Fig 7 — Collective Completion Times (CCT)
| Experiment | Best Spritz | Best Overall |
|------------|-------------|-------------|
| DF Allreduce Ring | Spray(u) 35.8ms | Valiant 35.0ms |
| DF Allreduce BF | Spray(w) 2.7ms | Spray(w) 2.7ms |
| DF Alltoall | Scout 12.5ms | Minimal 11.0ms |
| SF Allreduce Ring | Spray(w) 40.2ms | Spray(w) 40.2ms |
| SF Allreduce BF | Spray(u) 3.0ms | Spray(u) 3.0ms |
| SF Alltoall | Scout 22.9ms | UGAL-L 20.4ms |

### Fig 8 — Spritz speedup under 2% link failures (avg FCT)
| Experiment | DF speedup | SF speedup | Paper claim |
|------------|-----------|-----------|-------------|
| Adversarial | 5.4x | 4.9x | 2.5x-25.4x |
| Permutation | 7.7x | 4.3x | range |
| Allreduce Ring | 7.4x | 7.7x | |
| Allreduce BF | 11.0x | 9.2x | |
| Alltoall | 1.0x | 3.9x | |
| WebSearch | 2.2x | 1.8x | |

---

## Issues Encountered & Fixes
1. **System Python too old** — Python 3.6 lacks `bar_label()`. Fixed with Python 3.11 venv.
2. **OOM on first SLURM run** — 32GB insufficient for 64 parallel sims. Fixed: 256GB, 32 parallel.
3. **Missing subplots in `reproduce.sh`** — Only Fig 6 and partial Fig 7/8 were scripted. Added all missing plot commands.
4. **Adversarial naming mismatch** — DF uses `adv_i5_4MiB`, SF uses `adv_all_4MiB`. `plot.py` takes single experiment name. Fixed with custom combined plot script.
