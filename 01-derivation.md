# Derivation: FedAdaServer — diagonal-adaptive server steps and the $\kappa(D^{-1}A)$ contraction

This note derives the FedAdaServer update from vanilla FedAvg, formalizes the
round operator as an affine map on the parameter error, and shows *why* a
diagonal-adaptive (Adam/Yogi-style) server step should reduce the effective
condition number that governs the per-round contraction. The companion file
`proofs/fedavg-adaptive-diagonal-server-step.tex` gives the standalone
theorem statement and proof; this note is the working derivation with the
modeling assumptions made explicit and tied back to the toy in `prototype.py`.

## 1. FedAvg as a server step on an aggregated pseudo-gradient

FedAvg (McMahan et al. 2017, Algorithm 1) at round $t$ samples a client set
$S_t$, each client $k$ runs local SGD from the broadcast model $W_t$ and returns
$W_t^k$. The server forms the **Hájek-weighted aggregate**

$$ W_{t+1} \;=\; \sum_{k \in S_t} \frac{n_k}{m_t}\, W_t^k, \qquad m_t = \sum_{k\in S_t} n_k. $$

Equivalently, defining the **aggregated delta** $d_t := \sum_{k\in S_t}\frac{n_k}{m_t}(W_t^k - W_t)$,

$$ W_{t+1} \;=\; W_t + d_t. \tag{FedAvg} $$

Reddi et al. (2021, *Adaptive Federated Optimization*) reinterpret $-d_t$ as a
**server pseudo-gradient** $g_t := -d_t$: it points (in expectation) in the
direction the global model should move. Vanilla FedAvg is then the **identity
server optimizer**: $W_{t+1} = W_t - \eta_s g_t$ with $\eta_s = 1$. The identity
step inherits whatever conditioning the pseudo-gradient field has.

## 2. A linear model of one round

Assume a smooth global objective $F$ with minimizer $W^\star$ and Hessian
$H = \nabla^2 F(W^\star) \succ 0$. Let $e_t := W_t - W^\star$. Each round is local
GD plus client sampling. Linearizing the local-update-then-average map about
$W^\star$, the **expected** aggregated delta is linear in the error:

$$ \mathbb{E}[d_t \mid W_t] \;=\; -(I - M)\,e_t \;=\!:\; -B\,e_t, $$

where $M$ is the (expected) per-round local-progress operator (for one full-batch
local GD step of size $\eta$ on a quadratic, $M = I-\eta H$, so $B = \eta H$; for
$E$ local steps, $B = I-(I-\eta H)^E$, still SPD and **commuting with $H$**, hence
sharing $H$'s eigenbasis). Writing the realized delta as its mean plus zero-mean
client-sampling noise $\xi_t$,

$$ \boxed{\,d_t \;=\; -A\, e_t + \xi_t, \qquad A := B \succ 0,\quad \mathbb{E}[\xi_t\mid W_t]=0.\,} \tag{1} $$

The **eigenvalue spread of $A$ is the anisotropy** the toy bakes in: with
per-feature input stds $\sigma_j$ spanning $[1, 1/\mathrm{COND}]$, the softmax/Gaussian
data Hessian has $\mathrm{diag}(H)\propto \sigma_j^2$, so
$\kappa(A)\sim \mathrm{COND}^2 = 30^2$ for the quadratic part. FedAvg's error then
evolves as

$$ e_{t+1} \;=\; e_t + d_t \;=\; (I-A)e_t + \xi_t. \tag{FedAvg-error} $$

The deterministic part contracts at rate $\rho_{\text{FedAvg}} = \|I-A\|$.
Optimizing a single scalar step $\eta_s$ over the spectrum
$[\lambda_{\min},\lambda_{\max}]$ of $A$ gives the classic bound
$\rho^\star = 1 - \tfrac{2}{\kappa(A)+1} = 1 - \Theta(1/\kappa(A))$: **the flat
directions ($\lambda_{\min}$) gate the rate.** This is exactly the slow-flat-direction
pathology the gap targets.

## 3. The FedAdaServer step

Keep the Hájek aggregation (so $d_t$ is identical), but replace the identity
server step with a **diagonal-adaptive** update (Adam/Yogi), elementwise over the
flattened parameter vector:

$$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
   V_t = \beta_2 V_{t-1} + (1-\beta_2) g_t^{\odot 2}, $$
$$ W_{t+1} \;=\; W_t - \eta_s\, \frac{m_t}{\sqrt{V_t}+\epsilon}. \tag{FedAdaServer} $$

Take $\beta_1=0$ (no momentum, to isolate the preconditioner; this is the config
in `prototype.py`). Then $m_t = g_t = -d_t$ and the step is

$$ W_{t+1} = W_t + D_t^{-1} d_t, \qquad D_t := \tfrac{1}{\eta_s}\big(\operatorname{diag}(\sqrt{V_t})+\epsilon I\big). $$

So FedAdaServer is **FedAvg preconditioned by a diagonal $D_t^{-1}$**.

## 4. What $\sqrt{V_t}$ converges to

Plug (1) into the second-moment recursion. With $g_t = A e_t - \xi_t$,

$$ \mathbb{E}\big[g_t^{\odot 2}\big] = (A e_t)^{\odot 2} + \operatorname{diag}\!\big(\operatorname{Cov}(\xi_t)\big). $$

Two regimes:

* **Signal-dominated (transient).** While $\|e_t\|$ is large, $(Ae_t)^{\odot 2}$
  dominates and the per-coordinate second moment tracks the **squared gradient
  magnitude** of coordinate $j$, $\approx (A_{jj} e_{t,j})^2$. The EMA $\sqrt{V_t}$
  therefore approximates $|A e_t|$ coordinatewise, i.e. a diagonal that is *large*
  on steep coordinates and *small* on flat ones.
* **Noise-floor (near convergence).** Once $Ae_t \to 0$, $\sqrt{V_t}$ relaxes to
  the **sampling-noise floor** $\sqrt{\operatorname{diag}\operatorname{Cov}(\xi_t)}$,
  which is itself anisotropic (it scales with $\sigma_j$).

In both regimes $D_t$ is, **up to scale**, a diagonal that mimics the coordinate
scaling of $A$. The Adam preconditioner is the diagonal of $|A|^{1/2}$-type
information rather than $|A|$ itself, but the *ordering and spread* of coordinates
is preserved — that is all the conditioning argument needs.

## 5. The conditioning-reduction claim

Write the FedAdaServer error recursion (ignoring noise for the deterministic rate):

$$ e_{t+1} = (I - D_t^{-1}A)\, e_t. $$

If $D_t$ stabilizes to a diagonal $D$ whose entries are comparable to the
coordinate scales of $A$ (Section 4), then $D^{-1}A$ is **diagonally
preconditioned $A$**. For $A$ diagonal (the leading anisotropy of the toy, where
$A=\operatorname{diag}(a_j)$ with $a_j\propto\sigma_j^2$) and $D=\operatorname{diag}(d_j)$
with $d_j\propto\sqrt{a_j}$ (Adam's $\sqrt{V}$), we get
$D^{-1}A = \operatorname{diag}(a_j/\sqrt{a_j}) = \operatorname{diag}(\sqrt{a_j})$, hence

$$ \kappa(D^{-1}A) = \frac{\max_j \sqrt{a_j}}{\min_j \sqrt{a_j}} = \sqrt{\kappa(A)}. $$

The full $\epsilon$-regularized preconditioner only does *better* on the steep
coordinates, and at worst (when $\epsilon$ dominates) reduces to a uniform rescale
($D^{-1}A \to \epsilon^{-1}A$, same $\kappa$). So, with the matched server step,

$$ \boxed{\;\kappa(D^{-1}A) \;\le\; \kappa(A)\;,\quad \text{and for the diagonal toy } \kappa(D^{-1}A)\approx \sqrt{\kappa(A)}.\;} $$

The optimal-server-step contraction improves accordingly:

$$ \rho_{\text{FedAdaServer}} = 1 - \Theta\!\Big(\tfrac{1}{\kappa(D^{-1}A)}\Big)
   \;\le\; 1 - \Theta\!\Big(\tfrac{1}{\kappa(A)}\Big) = \rho_{\text{FedAvg}}. $$

For $\kappa(A)=\mathrm{COND}^2=900$ this predicts the gap going from $\sim 1/900$ to
$\sim 1/30$ per round — i.e. an **order-of-magnitude fewer rounds**, which is the
direction the experiment checks. (The theorem and a clean proof are in the `.tex`.)

## 6. Assumptions, and how they relate to FedAvg's heterogeneity term

* **(A1) Smooth, locally-strongly-convex $F$ near $W^\star$.** The toy is convex
  softmax regression, so $A\succ0$ holds globally enough for the linear model.
* **(A2) The round operator commutes with $H$** (true for full-batch local GD on a
  quadratic; approximately true for a few local steps). This is what lets us read
  $\kappa(A)$ off the data anisotropy.
* **(A3) Diagonal dominance of $A$.** Adam preconditions only the *diagonal*. The
  bound $\kappa(D^{-1}A)\approx\sqrt{\kappa(A)}$ is exact when $A$ is diagonal and
  degrades with off-diagonal coupling; the toy's anisotropy is largely
  coordinate-aligned, so this is a mild assumption here.
* **Heterogeneity / client drift.** Non-IID partitions inject a *bias* into
  $\mathbb{E}[d_t]$ (client drift), an additive term $b_t$ in (1) that the diagonal
  preconditioner does **not** remove — it rescales coordinates, it does not
  re-center them. So FedAdaServer attacks the **conditioning** axis of the gap,
  and is explicitly *orthogonal* to control-variate fixes (SCAFFOLD) that attack
  the **drift/bias** axis. This is why the idea is novel relative to FedAdam
  (purely empirical) and to aggregation/variance-reduction work.

## 7. What the experiment can and cannot confirm

The deliverable is the **bound**; the experiment only checks its *sign* on the
toy. Two measurable proxies:

1. **Rounds-to-target** should drop sharply (the $\rho$ improvement). *Strongly
   confirmed:* 62 → 4 rounds (15.5x), far beyond the predicted 2x.
2. **Per-coordinate equalization** of the effective step. *Partially confirmed:*
   the diagonal does make steps more equal (coefficient of variation 0.95 → 0.40
   at round 1), but the literal "5x" bar is missed — partly because $\epsilon=10^{-3}$
   exceeds most entries of $\sqrt{V_t}$ on this toy (so the preconditioner is
   $\epsilon$-dominated and acts closer to a well-scaled uniform step), and partly
   because the baseline's own per-coordinate spread *shrinks* as it slowly learns.
   See `findings.md` for the honest accounting.

The headline theoretical contribution — $\kappa(D^{-1}A)\le\kappa(A)$ giving a
strictly faster contraction — is consistent with the measured 15.5x round speedup;
the equalization sub-prediction was mis-calibrated and is reported as a partial miss.
