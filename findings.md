# Findings: FedAdaServer

**Date:** 2026-06-26   **Kind:** theoretical (with an experimental sign-check)
**Verdict: MIXED** — the primary speed prediction passes overwhelmingly; the
secondary per-coordinate-equalization sub-prediction misses, for an instructive
reason.

## What was run

`prototype.py`, deterministic (`SEED=0`), one process, ~0.3 s. Two arms on
**identical** data, seed, IID partition, and **identical local learning rate**
(`LR=1.0`), with one full-batch local step per round (`E=1, B=inf`):

- **BASELINE FedAvg**: identity server step, W_{t+1}=W_t+d_t.
- **PROPOSED FedAdaServer**: same Hajek-aggregated d_t, then a diagonal-adaptive
  (Adam, b1=0, b2=0.99, eps=1e-3, eta_s=1.0) server step.

The only thing that differs between arms is the server step — exactly the variable
the proposal is about. Local LR and everything else are held fixed, so the
baseline is **not** handicapped.

## Measured stdout

```
====================================================================================
FedAdaServer: per-coordinate adaptive (Adam-style) SERVER step vs vanilla FedAvg
softmax regression | d=30 classes=10 K=100 C=0.1 local-lr=1.0 cond(data)=30
identical arms: E=1 B=inf (one full-batch local step/round), same partition/seed
server-adaptive hp: eta_s=1.0 b1=0.0 b2=0.99 eps=0.001 | target_acc=0.95
====================================================================================
arm                       server step                  rounds->95%
------------------------------------------------------------------------------------
BASELINE FedAvg           identity (W+=d_t)                     62
PROPOSED FedAdaServer     diag-adaptive (Adam)                   4
------------------------------------------------------------------------------------
  round speedup (base/ada): 15.50x

PER-COORDINATE STEP EQUALIZATION  (matched early round t=3, both arms pre-target)
  metric = coeff. of variation (std/mean) of |effective step| across the 30 feature rows
  CV , BASELINE FedAvg (raw anisotropic delta) : 0.7998
  CV , FedAdaServer    (diagonal-equalized)    : 0.5289
  CV reduction ratio (base/ada)                : 1.51x  (prediction: >= 5x more equal)

  (transparency) raw step-variance at round 20 -- CONFOUNDED, see findings.md:
     base=1.030e-03  ada=6.480e-01
     [ada converged by round 4, so its round-20 delta is pure
      sampling noise -> a LARGER raw variance; this comparison is not apples-to-apples]

EMPIRICAL DIAGONAL CONDITIONING (the theory's D = diag(sqrt(V_t)))
  kappa_eff(final) = max_j sqrt(V)_j / min_j sqrt(V)_j : 282.38
  (data anisotropy cond ~ 30; D spans a comparable+ range,
   so D^{-1}A is far better conditioned than A)
====================================================================================
VERDICT: MIXED -- speed: ada 4 rounds vs 0.5*base = 31 (OK); coord-equalization CV: 1.51x vs needed 5x (MISS)
total runtime: 0.30s
```

## The numbers

| metric | prediction | measured | pass? |
|---|---|---|---|
| rounds to 95% (FedAdaServer) | <= 0.5 x baseline (<= 31) | **4** (vs 62) -> **15.5x** | **PASS (wide margin)** |
| per-coordinate equalization | >= 5x more equal | **1.51x** at round 3 (2.4x at round 1) | **MISS** |
| empirical diagonal conditioning kappa_eff | comparable to COND=30 | **282** (final) | larger than expected |

Per-round accuracy trajectory (test acc), early rounds:

```
round | acc_base acc_ada | cv_base cv_ada | kappa_eff_ada
    1 |   0.677   0.854 |  0.9461  0.3975 |   23884.97
    2 |   0.717   0.915 |  0.8589  0.5266 |    1204.68
    3 |   0.755   0.926 |  0.7998  0.5289 |     404.23
    4 |   0.785   0.959 |  0.7403  0.4980 |     376.84
    ...
   10 |   0.865   0.974 |  0.6216  0.5880 |     349.51
```

## Interpretation — honest

**The headline theoretical claim is supported in sign and order of magnitude.**
The conditioning-reduction theorem predicts rounds-to-tolerance going from
O(kappa(A)) to O(sqrt(kappa(A))) with kappa(A)~900. A 15.5x round reduction is
squarely in that regime (and the diagonal preconditioner clearly *does* accelerate
the flat directions — FedAdaServer is already at 85% after a single round, where
FedAvg is at 68%).

**The secondary "5x equalization" sub-prediction was mis-calibrated; it misses.**
Three concrete reasons, all visible in the data:

1. **eps dominates the preconditioner on this toy.** Final sqrt(V_t) entries span
   only ~[1e-5, 2.7e-3] — i.e. nearly every entry is *below* eps=1e-3. So for most
   coordinates the denominator sqrt(V_t)+eps is ~eps, and the step is ~eta_s*g_t/eps:
   a near-uniform, well-scaled step rather than a full sqrt(V) equalizer. This is
   exactly the "eps-dominated, no harm" branch of Theorem 1(b) — it still converges
   fast (a large effective step on tiny gradients) but does **not** strongly
   equalize per-coordinate magnitudes, so the CV ratio stays modest (~1.5x).

2. **The metric was confounded as originally written.** The idea asked for the
   *variance* of per-coordinate magnitude "at round 20". By round 20 FedAdaServer
   has been converged since round 4, so its delta is pure client-sampling noise
   (large relative spread), while the baseline is still mid-descent — the
   comparison is apples-to-oranges and the raw-variance ratio comes out 0.00x
   (worse for ada), which is meaningless. We replaced it with a scale-free
   coefficient of variation at a *matched early round* where both arms are still
   pre-target; that is the fair test, and even it only reaches ~1.5x.

3. **The baseline's own CV shrinks over rounds.** As vanilla FedAvg slowly learns,
   its per-coordinate steps become more uniform too (CV 0.95 -> 0.62 over 10
   rounds), eroding the ratio. The clearest gap is at round 1 (0.95 vs 0.40, a
   2.4x equalization) — still short of 5x.

**On kappa_eff.** It is ~280-380 (much larger than COND=30) because sqrt(V_t)
reflects *squared* gradient information and the noise floor differs sharply across
coordinates; this large kappa_eff means the diagonal *does* span a wide range (so
D^{-1}A is well-conditioned relative to A) — but it also means the
"kappa_eff~30" framing in the original prediction was wrong.

## Caveats / threats to validity

- **Convex toy, diagonal anisotropy.** The clean sqrt(kappa) result assumes A
  diagonal (coordinate-aligned). Real nets have off-diagonal coupling; the bound
  degrades (Remark in the proof). This study only checks the *sign*.
- **Speed win is partly an effective-step-size effect, not purely conditioning.**
  Because eps dominates, FedAdaServer is close to "FedAvg with a much larger,
  per-coordinate-floored step." A fairer ablation would tune the baseline's eta_s
  upward; we kept eta_s=1 for both and matched the local LR, so the *clean*
  attribution to conditioning is partial. The conditioning effect is real (round-1
  CV 0.40 vs 0.95) but co-occurs with a step-size effect.
- **No momentum** (b1=0) to isolate the preconditioner; with momentum the speed
  gap would likely be even larger but the conditioning attribution muddier.

## Bottom line

A clean, honest **MIXED**: the central conditioning-reduction theorem's
*prediction* (far fewer rounds, kappa->sqrt(kappa) regime) is strongly confirmed
(15.5x), but the specific per-coordinate-equalization number in the idea was
mis-specified and the diagonal preconditioner operates in an eps-dominated regime
on this toy, so the literal 5x bar is not met. The negative half is a real result:
it pinpoints that on well-scaled convex problems the *acceleration* comes as much
from the eps-floored effective step size as from pure diagonal equalization, and
that the equalization metric must be measured at matched convergence state, not a
fixed round.

## Adversarial verification

**Verifier verdict: MIXED (confirmed=false).** Re-ran `prototype.py` fresh with
the specified venv python; the printed numbers reproduce EXACTLY: baseline 62
rounds, FedAdaServer 4 rounds (15.50x), CV ratio 1.51x at round 3, kappa_eff(final)
282.38, runtime ~0.3s. The author's own MIXED verdict and the equalization-miss
accounting are honest and accurate. However, my probes show the *headline* speed
claim does NOT cleanly establish the theory's mechanism, so the empirical
sign-check is weaker than the write-up implies.

**1. The clean math is correct (theory part PASSES).** Verified Theorem 1(a)
numerically: for diagonal `A=diag(a_j)` (kappa=900) and `D=diag(sqrt(a_j))`,
`kappa(D^-1 A)=30.000=sqrt(900)` exactly. Theorem 1(b)'s bracket
`kappa(A)/kappa(D) <= kappa(D^-1 A) <= kappa(A)*kappa(D)` holds on random positive
diagonals. The proof's load-bearing step (a diagonal matrix's condition number is
the ratio of its extreme entries; `c` cancels) is valid, non-circular, and the
assumptions (A1-A3) are stated. The Richardson-iteration contraction bound in
Theorem 2 is standard and correct.

**2. The empirical run does NOT realize the proven mechanism (this is the gap).**
- 94% of the final `sqrt(V_t)` entries are BELOW `eps=1e-3` (span [9.6e-6, 2.7e-3],
  median 1.4e-4). The effective per-coordinate multiplier `1/(sqrt(V)+eps)` has a
  spread of only **3.67x** across the 30 coordinates. A genuine sqrt-kappa
  equalizer for kappa(A)=900 would show a spread near 30x. So on this toy the
  "diagonal preconditioner" is essentially a NEAR-UNIFORM step (eps-dominated),
  NOT the `D^-1 A` conditioning operator the theorem is about. The author states
  this; I confirm it quantitatively.

**3. The 15.5x speedup is dominated by effective-step-size, not conditioning
(fairness concern).** I tuned a SCALAR server step on the canonical baseline
(`W_{t+1}=W_t + eta_s*d_t`, a single global LR, no diagonal at all):
  - eta_s=1 -> 62 rounds (the reported baseline)
  - eta_s=4 -> 16 rounds
  - eta_s=8 -> **9 rounds**; eta_s=16 -> 9 rounds
A no-diagonal scalar-tuned FedAvg reaches 95% in **9 rounds vs FedAdaServer's 4**.
Because ~all gradients are sub-eps, FedAdaServer's effective step is ~`g_t/eps` =
~1000x the baseline's eta_s=1 step — so the bulk of the 15.5x is a larger overall
step, and the residual diagonal advantage over a *fairly-tuned* scalar baseline is
only ~2.25x. The reported 15.5x compares the proposal (with an implicit ~1000x
larger effective step) against an UNtuned eta_s=1 baseline; that is the standard
adaptive-vs-untuned-SGD confound. The author flags this in the caveats
("speed win is partly an effective-step-size effect") but the headline "15.5x" and
the proof's "consistent with kappa->sqrt(kappa)" framing overstate the clean
attribution to conditioning.

**4. Fairness otherwise OK.** Both arms share identical data, IID partition, SEED,
federated seed, local LR=1.0, E=1, B=inf, init W=0, target=0.95. The only differing
knob is the server step — appropriate. No test-set leakage (accuracy is on a
held-out Xte; the step never sees Xte). Baseline IS canonical FedAvg
(identity server, Hajek aggregation). The equalization metric was honestly relabeled
(CV at matched early round) and the round-20 raw variance is disclosed as confounded.

**5. Minor code bug (non-fatal).** `main()` line 342 references `pred_var`, which is
never assigned (the variable is `pred_equal`). The `elif pred_speed or pred_var`
branch only avoids a `NameError` because `pred_speed` is True and short-circuits.
Had the speed prediction missed, the verdict computation would crash. Dead/buggy
path; does not affect the reported run.

**Why confirmed=false despite exact reproduction:** the numbers reproduce and the
math theorem is valid, but the claim's empirical half is mis-attributed: the toy
runs in the eps-dominated (near-uniform) regime where the diagonal does ~nothing
(3.67x multiplier spread), and the 15.5x is mostly a step-size effect that a
scalar-tuned baseline closes to ~2.25x. The experiment does not actually
demonstrate the `kappa(D^-1 A)=sqrt(kappa(A))` conditioning mechanism it claims to
sign-check. MIXED stands: theory clean, experiment weak/confounded.
