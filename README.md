# FedAdaServer: per-coordinate adaptive (Adam-style) server steps vs FedAvg

[![Render LaTeX](https://github.com/pleyva2004/fedavg-adaptive-diagonal-server-step/actions/workflows/render.yml/badge.svg)](https://github.com/pleyva2004/fedavg-adaptive-diagonal-server-step/actions/workflows/render.yml)

> A forward-looking mathematical extension to **FedAvg** ([arXiv:1602.05629](https://arxiv.org/abs/1602.05629)). Companion to the [parent study](https://github.com/pleyva2004/communication-efficient-learning-of-deep-networks).

A self-contained mini-study extending **FedAvg** (McMahan et al. 2017,
[arXiv:1602.05629](https://arxiv.org/abs/1602.05629)) along the *server-side
optimization & acceleration* lens.

## The FedAvg gap

FedAvg's server step is the **identity**: it just adds the Hájek-weighted
aggregated delta, $W_{t+1}=W_t+d_t$. That step inherits the **raw anisotropy** of
the global objective — the slow flat directions that the toy mimics with
`COND=30`. Flat coordinates converge orders of magnitude slower than steep ones,
and the original paper has no per-coordinate scaling and no convergence theory to
expose this.

## The mechanism

Keep the Hájek aggregation, but treat $g_t=-d_t$ as a **server pseudo-gradient**
and replace the identity step with a **diagonal-adaptive (Adam/Yogi-style)** update,
elementwise:

$$ m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad V_t=\beta_2 V_{t-1}+(1-\beta_2)g_t^{2},\quad
   W_{t+1}=W_t-\eta_s\frac{m_t}{\sqrt{V_t}+\epsilon}. $$

With $\beta_1=0$ this is exactly **FedAvg preconditioned by a diagonal**
$D_t^{-1}\propto(\diag(\sqrt{V_t})+\epsilon I)^{-1}$.

## The math result (deliverable)

Model one round as an affine map on the error, $d_t=-A e_t+\xi_t$, where $A\succ0$
is the round operator (its eigenvalue spread = the `COND=30` anisotropy) and
$\xi_t$ is zero-mean client-sampling noise. Then (see `01-derivation.md`,
`proofs/`):

- $\sqrt{V_t}$ stabilizes to a diagonal tracking the **coordinate scales** of $A$;
- $\boxed{\kappa(D^{-1}A)\le\kappa(A)}$, and for the diagonal toy
  $\kappa(D^{-1}A)=\sqrt{\kappa(A)}$;
- the per-round contraction improves from $1-\Theta(1/\kappa(A))$ to
  $1-\Theta(1/\sqrt{\kappa(A)})$ → rounds-to-tolerance go from $O(\kappa)$ to
  $O(\sqrt{\kappa})$.

This conditioning-reduction theorem is the new contribution; it is **orthogonal**
to control-variate/drift fixes (it rescales coordinates, it does not re-center the
heterogeneity bias).

## The MEASURED result

On the convex softmax toy ($\kappa(A)\approx \mathrm{COND}^2=900$), identical
data/seed/partition/local-LR, one full-batch local step per round (`E=1, B=inf`):

| arm | server step | rounds to 95% |
|---|---|---:|
| BASELINE FedAvg | identity ($W{+}{=}d_t$) | **62** |
| PROPOSED FedAdaServer | diagonal-adaptive (Adam) | **4** |

**15.5x fewer rounds** — far beyond the predicted 2x, and consistent in
order-of-magnitude with $\kappa\to\sqrt\kappa$.

**VERDICT: MIXED.** The headline speed prediction passes overwhelmingly; the
secondary "per-coordinate equalization ≥ 5x" sub-prediction **misses** (measured
~1.5x at round 3, ~2.4x at round 1). The miss is informative: the practical
$\epsilon=10^{-3}$ is *larger* than nearly every entry of $\sqrt{V_t}$ on this toy,
so the preconditioner runs in the $\epsilon$-dominated regime (a well-scaled
near-uniform step) rather than as a full $\sqrt{V}$ equalizer — yet that regime is
exactly the "no harm" case of the theorem and still delivers the big round win. See
`findings.md` for the honest accounting.

## How to run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python prototype.py            # < 1s; prints the comparison table + VERDICT
```

To compile the proof:

```bash
cd proofs && latexmk -pdf fedavg-adaptive-diagonal-server-step.tex
```

## Files

- `metadata.json` — study metadata
- `01-derivation.md` — rigorous derivation: FedAvg update → FedAdaServer → why $\kappa(D^{-1}A)\le\kappa(A)$, assumptions, relation to the heterogeneity/drift term
- `proofs/fedavg-adaptive-diagonal-server-step.tex` — standalone compilable proof of the conditioning-reduction theorem and the contraction-rate corollary
- `prototype.py` — self-contained, deterministic; baseline FedAvg vs FedAdaServer on identical data; prints table + `VERDICT`; exposes `measure()`
- `findings.md` — measured numbers, verdict, honest caveats
- `requirements.txt` — `numpy>=2.0`
