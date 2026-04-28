"""
Power Analysis for Multi-Location IOR Experiment
=================================================
Design: 1 SOA (~300ms), 6 equivalent horizontal locations, detection task
Within-subjects repeated-measures design.

Hypothesis: IOR (valid > invalid RT) occurs uniformly across all 6 locations.
The 6 locations test whether IOR generalizes across multiple spatial positions,
with no predicted eccentricity modulation.

Analyses:
  1. Overall IOR: valid vs invalid (paired t-test, collapsed across locations)
  2. IOR per location: simple effects at each of the 6 cue positions
  3. Location x Validity interaction: does IOR magnitude vary across 6 positions?
     (expected null — want to show IOR is uniform)

Uses Monte Carlo simulation (5000 iterations per condition).
"""

import numpy as np
from scipy import stats

np.random.seed(42)

# --------------------------------------------------------------
# Parameters
# --------------------------------------------------------------
# 6 positions: 0=far-left, 1=mid-left, 2=near-left,
#              3=near-right, 4=mid-right, 5=far-right
# All assumed equivalent — same IOR effect at each location.

N_LOCATIONS = 6
MEAN_RT_BASE = 350            # ms, baseline (invalid/uncued) RT
IOR_EFFECT = 20               # ms, RT cost at cued location (same everywhere)

SD_WITHIN = 55                # ms, within-subject trial-to-trial SD
SD_BETWEEN = 40               # ms, between-subject SD of overall RT
SD_EFFECT_BETWEEN = 8         # ms, between-subject SD of IOR magnitude

N_SIMULATIONS = 5000
ALPHA = 0.05

PARTICIPANT_RANGE = [10, 15, 20, 25, 30, 40, 50]
REPS_PER_CELL_RANGE = [5, 8, 10, 15, 20]


# --------------------------------------------------------------
# Simulation
# --------------------------------------------------------------

def simulate_one(n_participants, reps_per_cell):
    """
    Simulate one experiment.

    Each participant: 6 cue positions x 2 validities x reps_per_cell trials.
    IOR effect is identical at all 6 locations (uniform model).

    Returns p-values for each analysis.
    """
    # Data: participants x 6 locations x 2 (0=valid, 1=invalid)
    data = np.zeros((n_participants, N_LOCATIONS, 2))

    for s in range(n_participants):
        subj_offset = np.random.normal(0, SD_BETWEEN)
        subj_ior = np.random.normal(IOR_EFFECT, SD_EFFECT_BETWEEN)
        subj_ior = max(subj_ior, 0)

        for loc in range(N_LOCATIONS):
            # Valid trials (cued): RT = base + offset + IOR + noise
            rts_valid = (MEAN_RT_BASE + subj_offset + subj_ior
                         + np.random.normal(0, SD_WITHIN, reps_per_cell))
            data[s, loc, 0] = np.mean(rts_valid)

            # Invalid trials (uncued): RT = base + offset + noise
            rts_invalid = (MEAN_RT_BASE + subj_offset
                           + np.random.normal(0, SD_WITHIN, reps_per_cell))
            data[s, loc, 1] = np.mean(rts_invalid)

    # IOR per location per participant
    ior_per_loc = data[:, :, 0] - data[:, :, 1]  # positive = IOR

    # -- Analysis 1: Overall IOR (collapsed across locations) --
    mean_valid = np.mean(data[:, :, 0], axis=1)
    mean_invalid = np.mean(data[:, :, 1], axis=1)
    t_stat, p_val = stats.ttest_rel(mean_valid, mean_invalid)
    p_overall = p_val / 2 if t_stat > 0 else 1 - p_val / 2

    # -- Analysis 2: IOR at each location (simple effects) --
    p_per_loc = []
    for loc in range(N_LOCATIONS):
        t, p = stats.ttest_rel(data[:, loc, 0], data[:, loc, 1])
        p_loc = p / 2 if t > 0 else 1 - p / 2
        p_per_loc.append(p_loc)

    # -- Analysis 3: Location x Validity interaction --
    # RM-ANOVA on IOR difference scores across 6 locations
    # Under the null (uniform IOR), this should NOT be significant
    n = n_participants
    k = N_LOCATIONS
    grand_mean = np.mean(ior_per_loc)
    cond_means = np.mean(ior_per_loc, axis=0)
    subj_means = np.mean(ior_per_loc, axis=1)

    ss_cond = n * np.sum((cond_means - grand_mean) ** 2)
    ss_subj = k * np.sum((subj_means - grand_mean) ** 2)
    ss_total = np.sum((ior_per_loc - grand_mean) ** 2)
    ss_error = ss_total - ss_cond - ss_subj

    df_cond = k - 1
    df_error = (n - 1) * (k - 1)
    ms_cond = ss_cond / df_cond
    ms_error = ss_error / df_error if df_error > 0 else 1e10
    f_stat = ms_cond / ms_error if ms_error > 0 else 0
    p_interaction = 1 - stats.f.cdf(f_stat, df_cond, df_error)

    return p_overall, p_per_loc, p_interaction


def run_power(n_participants, reps_per_cell, n_sims=N_SIMULATIONS):
    """Run n_sims simulations and compute power for each analysis."""
    sig_overall = 0
    sig_all_locs = 0          # all 6 locations individually significant
    sig_min_loc = [0] * N_LOCATIONS
    sig_interaction = 0       # false positive rate (should be ~5%)

    for _ in range(n_sims):
        p_ov, p_locs, p_int = simulate_one(n_participants, reps_per_cell)

        if p_ov < ALPHA:
            sig_overall += 1
        if p_int < ALPHA:
            sig_interaction += 1

        all_sig = True
        for loc in range(N_LOCATIONS):
            if p_locs[loc] < ALPHA:
                sig_min_loc[loc] += 1
            else:
                all_sig = False
        if all_sig:
            sig_all_locs += 1

    power_overall = sig_overall / n_sims
    power_all_locs = sig_all_locs / n_sims
    power_worst_loc = min(sig_min_loc[loc] / n_sims for loc in range(N_LOCATIONS))
    false_pos_interaction = sig_interaction / n_sims
    return power_overall, power_worst_loc, power_all_locs, false_pos_interaction


def main():
    print("=" * 80)
    print("POWER ANALYSIS: Multi-Location IOR (6 equivalent positions)")
    print("=" * 80)

    print(f"\nDesign: 1 SOA | 6 horizontal locations | Detection task")
    print(f"Hypothesis: IOR is uniform across all 6 locations (no eccentricity effect)")
    print(f"\nAssumed parameters:")
    print(f"  IOR effect:          {IOR_EFFECT} ms (same at all locations)")
    print(f"  Baseline RT:         {MEAN_RT_BASE} ms")
    print(f"  Within-subject SD:   {SD_WITHIN} ms")
    print(f"  Between-subject SD:  {SD_BETWEEN} ms")
    print(f"  IOR variability:     {SD_EFFECT_BETWEEN} ms (between subjects)")
    print(f"  Alpha:               {ALPHA} (one-tailed for IOR, two-tailed for interaction)")
    print(f"  Simulations:         {N_SIMULATIONS}")

    print(f"""
Columns:
  Overall   = Valid vs invalid collapsed (paired t-test) -- main IOR effect
  Worst loc = Power at the weakest single location (conservative)
  All 6 sig = Probability that IOR is significant at ALL 6 locations
  FP int    = False positive rate for Location x Validity interaction
              (should be ~5% since IOR is truly uniform)
""")

    total = len(PARTICIPANT_RANGE) * len(REPS_PER_CELL_RANGE)
    print(f"Running {total} conditions x {N_SIMULATIONS} simulations...\n")

    all_results = []

    for n_part in PARTICIPANT_RANGE:
        print(f"{'-' * 80}")
        print(f"  N = {n_part} participants")
        print(f"{'-' * 80}")
        print(f"  {'Reps':>4} | {'Trials':>6} | {'Overall':>7} | "
              f"{'Worst loc':>9} | {'All 6 sig':>9} | {'FP int':>6}")
        print(f"  {'-'*4} | {'-'*6} | {'-'*7} | "
              f"{'-'*9} | {'-'*9} | {'-'*6}")

        for reps in REPS_PER_CELL_RANGE:
            total_trials = 2 * N_LOCATIONS * reps
            pw_ov, pw_worst, pw_all, fp_int = run_power(n_part, reps)
            all_results.append((n_part, reps, total_trials,
                                pw_ov, pw_worst, pw_all, fp_int))

            print(f"  {reps:>4} | {total_trials:>6} | "
                  f"{pw_ov:>6.0%} | "
                  f"{pw_worst:>8.0%} | "
                  f"{pw_all:>8.0%} | "
                  f"{fp_int:>5.1%}")

    # -- Summary --
    print(f"\n{'=' * 80}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'=' * 80}")

    print("""
Your key analyses:
  1. Overall IOR (valid > invalid): This is easy to power.
  2. IOR at each location: Harder, since each location has fewer trials.
     "All 6 sig" = the probability you detect IOR at EVERY location.
     This is what you need to convincingly show IOR is present across
     all 6 positions.
  3. Location x Validity interaction: You expect this to be NULL (uniform IOR).
     "FP int" should hover around 5%. If it's inflated, your design has a
     specificity problem. If it's ~5%, your design is well-calibrated.
""")

    # Configs where all 6 locations are individually significant >= 80%
    good = [(n, r, t, ov, wl, a6, fp) for n, r, t, ov, wl, a6, fp
            in all_results if a6 >= 0.80]
    if good:
        print("Configs where IOR is significant at ALL 6 locations >= 80% of the time:")
        print(f"  {'N':>4} | {'Reps':>4} | {'Trials':>6} | "
              f"{'Overall':>7} | {'All 6':>5} | {'FP int':>6}")
        print(f"  {'-'*4} | {'-'*4} | {'-'*6} | "
              f"{'-'*7} | {'-'*5} | {'-'*6}")
        for n, r, t, ov, wl, a6, fp in good:
            print(f"  {n:>4} | {r:>4} | {t:>6} | "
                  f"{ov:>6.0%} | {a6:>4.0%} | {fp:>5.1%}")
        best = min(good, key=lambda x: x[0] * x[2])
        print(f"\n  >> Recommended minimum: {best[0]} participants, "
              f"{best[1]} reps/cell ({best[2]} trials), "
              f"all-6 power = {best[5]:.0%}")
    else:
        print("  No config reached 80% for all 6 locations simultaneously.")
        print("  Consider more trials or participants.")

    # Current design
    print(f"\n{'-' * 80}")
    print("YOUR CURRENT DESIGN: 10 reps/cell (120 trials)")
    cur = [(n, r, t, ov, wl, a6, fp) for n, r, t, ov, wl, a6, fp
           in all_results if r == 10]
    if cur:
        print(f"  {'N':>4} | {'Overall':>7} | {'Worst loc':>9} | "
              f"{'All 6 sig':>9} | {'FP int':>6}")
        for n, r, t, ov, wl, a6, fp in cur:
            print(f"  {n:>4} | {ov:>6.0%} | {wl:>8.0%} | "
                  f"{a6:>8.0%} | {fp:>5.1%}")
    print(f"{'-' * 80}")


if __name__ == "__main__":
    main()
