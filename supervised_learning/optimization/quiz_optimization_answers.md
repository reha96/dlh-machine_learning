# Optimization — Answer Key & Explanations (24Q, Expanded)

> Separate file — keep folded while you attempt `quiz_optimization_mcq.md`.
> Grounding: `RESOURCES.md` summaries + AI Book notebook
> (*Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow,
> 3rd ed.*, ID `notebook:rxx4byfysdltkrffq02y`, evaluation
> `source:zbqd84whl6a4kbcc1y9x`, captured 2026-08-30, session
> `a3706952`) + `blog.md` 7-error fixes.
> **Expansion:** 20Q → 24Q keeps Q1–20 intact and adds Q21–24 as the
> four deepest notebook probes where 20Q was indirect. Quick key and
> per-question notes below cover all 8 notebook MCQs; mapping is in the
> MCQ header.

**Quick key:** 1C · 2B · 3D · 4A · 5C · 6B · 7D · 8A · 9C · 10B · 11D · 12A · 13C · 14B · 15D · 16A · 17C · 18B · 19D · 20A · 21A · 22C · 23B · 24D
*(Correct positions randomized ≈ 6× each letter: A=6, B=6, C=6, D=6.
Stems are high → low within each topic block — see header in MCQ file.)*

**Scoring (undergrad, 24Q):** 22–24 excellent · 19–21 solid (review missed
topics) · 15–18 gaps — re-read `RESOURCES.md:448-478` Quiz Hooks ·
<15 redo with notebook and `RESOURCES.md:30-76, 100-113`.

---

### Q1. Feature scaling — why scale? → **C**
**Why C:** Different ranges stretch cost contours; weights live on
different scales so GD zig-zags and needs tiny `alpha`
`RESOURCES.md:64-68,223`. Scaling to zero-mean unit-variance rounds
the bowl so steps point toward the minimum.
Notebook `source:zbqd84whl6a4kbcc1y9x` Q2: correct is elongated bowl,
not exploding gradients.
**Why not:** A claims GD is scale-invariant like trees
`RESOURCES.md:57-58` (trees split on thresholds, GD uses geometry);
B confuses optimization with heteroskedasticity diagnostics; D
inverts the effect.
**Econ bridge:** `income` in dollars vs `schooling` in years — like
PPP vs local-currency GDP on one plot without rescaling.
**Blog fix:** `blog.md:14` purpose "avoid exploding gradient" → fix to
elongated-bowl / narrow valley intuition.

### Q2. Feature scaling — standardization vs min-max → **B**
**Why B:** Standardization `x'=(x−mean)/std` → mean 0, variance 1,
the default for SVM, logistic regression and nets
`RESOURCES.md:42-43,453-454`. Min-max `x'=(x−min)/(max−min)` → [0,1]
`RESOURCES.md:40,454`.
**Distractors:** A swaps formulas and claims robustness to outliers;
C swaps the other way and claims min-max resists outliers — opposite,
robust scaling uses median/IQR `RESOURCES.md:44`. D invents
`gamma·x+beta` as min-max — that is batch norm's affine step
`RESOURCES.md:73`.
**Blog fix:** `blog.md:13` wrote normalization as `((X*gamma)+beta)`
— that's batch norm, not min-max. Notebook Q1 says Both min-max
and standardization are valid; this Q tests the paired formulas.

### Q3. Feature scaling — practice, heavy tails and train/test → **D**
**Why D:** Use **same** `mean, std` from train on test
`RESOURCES.md:225,457`; log-transform very skewed features (population,
GDP) first so skew does not dominate the scaling.
**Why not:** A leaks test info by recomputing on test; B mixes methods
across splits; C claims input scaling is redundant if batch norm
follows — false, BN helps hidden layers but input scaling still rounds
the input-space bowl `RESOURCES.md:228`.
**Econ bridge:** Like using CPI base-year weights for both estimation
and forecast samples — do not rebase on the forecast sample.

### Q4. Batch norm — purpose and intuition (high-level) → **A**
**Why A:** BN pins each layer's pre-activation mean/variance per
mini-batch, decoupling layers so deeper layers see stable inputs
(taming covariate shift, black-cat→colored-cat analogy
`RESOURCES.md:394-396`), allowing larger `alpha` and very deep nets
`RESOURCES.md:363-366,389`; noisy batch stats add mild dropout-like
regularization `RESOURCES.md:397-399`. Notebook Q3 correction: BN
does this every mini-batch, not at intervals.
**Why not:** B is learning-rate decay, not BN — the exact confusion in
`blog.md:17`. C claims BN deletes `gamma, beta` to shrink the model
(opposite); D claims BN always hurts wall-clock time, ignoring faster
convergence offsetting per-epoch cost.

### Q5. Batch norm — core mechanics → **C**
**Why C:** Training: `mu, sigma2` per batch → `z_norm=(z−mu)/
sqrt(sigma2+eps)` → `z_tilde=gamma·z_norm+beta` → `a=g(z_tilde)`; bias
`b` cancels under mean subtraction so `beta` replaces it
`RESOURCES.md:71-76,369-387,473`; `gamma, beta` shape `(n^{[l]},1)`
`RESOURCES.md:386`. Notebook Q3: this happens at every training step
per mini-batch.
**Why not:** A uses min-max `(z−min)/(max−min)` — wrong norm; B
normalizes gradients per epoch — that's LR scheduling
`blog.md:17` error; D restricts BN to raw inputs only
`RESOURCES.md:365` says hidden layers are the point.

### Q6. Batch norm — test time and details → **B**
**Why B:** At inference with one example there is no batch, so use the
exponentially weighted running averages of `mu, sigma2` collected
during training together with learned `gamma, beta`
`RESOURCES.md:410-417,474`; frameworks store them by default
`RESOURCES.md:415`.
**Why not:** A recomputes from one example and zeros `gamma, beta`;
C skips normalization; D overwrites `gamma, beta` with test-set stats
— they are learned parameters, not recomputed per test set.

### Q7. Mini-batch — why it is the default (intuition) → **D**
**Why D:** Frequent progress before epoch ends (survey-sample analogy),
vectorization within the batch, lower variance than SGD, cheaper and
more memory-friendly than full-batch GD `RESOURCES.md:122-128`.
**Why not:** A describes batch GD (exact but slow);
B claims hyperparameters become irrelevant (false); C guarantees escape
from every saddle without momentum/Adam (overclaim).
**Econ bridge:** Census (batch) precise but slow; one interview at a
time (SGD) noisy; stratified survey (mini-batch) balances precision,
speed and cost.

### Q8. Mini-batch — mechanics vs batch/SGD → **A**
**Why A:** Batch GD: one update per epoch (cost over all `m`); SGD: one
per example (noisy, loses vectorization); mini-batch e.g. 256: one per
batch (≈19,500 updates/epoch for 5M), vectorized inside each batch
`RESOURCES.md:234-242,458`. Notebook Q4: small random subset each
iteration.
**Why not:** B equates mini-batch with batch normalization — the exact
mix-up in `blog.md:20`; C swaps the counts; D claims same number of
updates for all three.

### Q9. Mini-batch — cost curve and sizing details → **C**
**Why C:** Batch GD cost must decrease each iteration or `alpha` is too
large; mini-batch cost trends down but is noisy (some batches harder)
`RESOURCES.md:248-249,254`; size guide `m ≤ 2000 → batch GD; else
64–512, power of 2, fits GPU` `RESOURCES.md:255-256,460`; 32 is a good
default (Bengio) and the small-batch study favors ≤32
`RESOURCES.md:26-27,127-128`.
**Why not:** A claims mini-batch must strictly decrease and size fixed
at 1000; B says SGD is smoothest; D claims batch size is learned by GD
like `W, b`.

### Q10. EWMA — what beta controls → **B**
**Why B:** `V_t=beta·V_{t-1}+(1−beta)·theta_t`, window ≈ `1/(1−beta)`
`RESOURCES.md:139,462`; `beta=0.9 → ~10`, `0.98 → ~50, smoother but
laggy` `RESOURCES.md:264-271`; weights `(1−beta)·beta^k`
`RESOURCES.md:278-281`; one scalar memory `RESOURCES.md:285`.
**Why not:** A is `alpha` (learning rate); C claims `beta` is a count of
stored batches; D claims `beta=0.9` disables averaging.

### Q11. Momentum — update and ravine intuition → **D**
**Why D:** `v_dW=beta·v_dW+(1−beta)·dW; W←W−alpha·v_dW` (same for `b`)
`RESOURCES.md:107,305-315,464`; damps vertical oscillation, accumulates
horizontal progress — ball downhill with friction `beta`
`RESOURCES.md:308`, so larger `alpha` is safe on elongated bowls.
Notebook Q5: exponentially decayed past gradients, accumulation in
consistent direction.
**Why not:** A is RMSProp `v_dW=dW/sqrt(s+eps)` `RESOURCES.md:156,465`;
B adds weight decay `beta·W`; C claims momentum normalizes inputs to
[0,1] — that's feature scaling confusion.

### Q12. Momentum — hyperparameters and bias correction → **A**
**Why A:** `beta=0.9` default (≈10-step average) `RESOURCES.md:312`;
bias correction `v/(1−beta^t)` exists `RESOURCES.md:292-294,463` but is
usually skipped for momentum `RESOURCES.md:298,313`; variant
`v=beta·v+dW` is equivalent up to rescaling `alpha` but then changing
`beta` also rescales the step `RESOURCES.md:315`; works with batch
**and** mini-batch `RESOURCES.md:311`.
**Why not:** B swaps `beta=0.999` (that's Adam's `beta2`) and calls
correction mandatory; C swaps `epsilon` for `beta`; D restricts
momentum to batch GD only.

### Q13. RMSProp — purpose and intuition (high-level) → **C**
**Why C:** Per-parameter adaptive rates: directions with large
oscillating gradients build a large running average of squared gradients
and get smaller effective steps; quiet directions keep larger steps —
curbing vertical bounce while preserving flat-axis progress
`RESOURCES.md:158-159`. This fixes Adagrad's monotonically growing sum
that collapses the rate `RESOURCES.md:109-110,156-159`.
**Why not:** A keeps one global rate; B is Adagrad's sum without decay;
D is batch norm's `gamma, beta` — the confusion noted in `blog.md:11`
and `RESOURCES.md:73`.

### Q14. RMSProp — core mechanics → **B**
**Why B:** `s_dW=beta·s_dW+(1−beta)·dW^2` (element-wise square),
`W←W−alpha·dW/sqrt(s_dW+eps)` `RESOURCES.md:156,322,465`; `s_dW` is the
EWMA of squared gradients (sign discarded) `RESOURCES.md:153-156`;
`beta≈0.9`, `eps≈1e-8` for stability `RESOURCES.md:329`.
Notebook Q6: second (uncentered) mean of squared gradients only.
**Why not:** A tracks mean `dW` without squaring — the `blog.md:27`
error claiming 1st+2nd moments; C is Adagrad's forever-growing sum; D
is batch norm's affine step.
**Blog fix:** `blog.md:27` said RMSProp uses "mean and variance (1st
and 2nd moments)" — only 2nd.
**Econ bridge:** Like GARCH variance `sigma_t^2=omega+alpha·eps_{t-1}^2+
beta·sigma_{t-1}^2` — decayed squares set the next scale.

### Q15. RMSProp — details, econ bridge and pros/cons → **D**
**Why D:** Decayed average of squares weights recent volatility most —
RMSProp shrinks steps where recent squared gradients were large
(volatile regressor) and keeps steps where gradients were flat, like
weighting assets by realized volatility `RESOURCES.md:158-159`.
Intro cost: extra `beta≈0.9` rarely tuned; occasional generalization
gap vs SGD+momentum (Bushaev medium article via notebook).
**Why not:** A equal-weights forever (that's Adagrad); B requires
min-max before squaring; C tracks signed mean, not squares.

### Q16. Adam — purpose and behavior (high-level) → **A**
**Why A:** Adam = momentum (first moment) + RMSProp (second moment) with
bias correction; step bounded by `alpha` and largely invariant to
gradient rescaling, helping on plateaus, saddles and sparse features
`RESOURCES.md:111,173,338,343,466`; correction matters early because
`m, v` start at 0 `RESOURCES.md:174`. Notebook Q7: combines RMSProp
with Momentum.
**Why not:** B uses only second moment; C is inverse-time LR decay;
D is batch norm `gamma·z_norm+beta` — name collision only.
**Blog fix:** `blog.md:30` said "RMS+momentum with bias correction" —
true but incomplete without both moments and `eps`.

### Q17. Adam — core mechanics and formulas → **C**
**Why C:** `m=beta1·m+(1−beta1)·g; v=beta2·v+(1−beta2)·g^2;
m_hat=m/(1−beta1^t); v_hat=v/(1−beta2^t);
theta←theta−alpha·m_hat/(sqrt(v_hat)+eps)` `RESOURCES.md:111,173,338,
466`; name = Adaptive Moment Estimation (1st + 2nd moments)
`RESOURCES.md:343`.
**Why not:** A uses `m=beta·m+g^2` with no correction; B is linear LR
decay `alpha·g/(1+beta·t)`; D is batch norm affine + plain GD.

### Q18. Adam — defaults, cons and modern fixes → **B**
**Why B:** Defaults `beta1=0.9, beta2=0.999, eps=1e-8` (Keras `1e-7`
variant) `RESOURCES.md:111,341-342,467`; tune `alpha` first
`RESOURCES.md:342`; `E[m_t]=(1−beta1^t)E[g]` so divide correction
`RESOURCES.md:174`; step bounded by `alpha`, invariant to scale
`RESOURCES.md:175`; downside: can generalize worse than SGD+momentum
(Wilson et al. `RESOURCES.md:178`), Reddi counterexample — forgetting
rare grads, AMSGrad keeps max `v` `RESOURCES.md:176`; AdamW decouples
weight decay `RESOURCES.md:177`; SWATS switches Adam→SGD
`RESOURCES.md:178`; original L2 ≠ weight decay for Adam.
**Why not:** A swaps `beta1/beta2` and correction; C claims only cost
is `gamma, beta` memory and L2 equals weight decay; D claims Adam
always diverges `RESOURCES.md:344` says Ng found it robust.

### Q19. LR decay — why and when vs adaptives (high-level) → **D**
**Why D:** Fixed `alpha` makes mini-batch wander near the minimum
`RESOURCES.md:350-353`; decaying `alpha` gives large early steps (fast
progress, escape plateaus) and small late steps (settle)
`RESOURCES.md:351-353`; but a well-tuned fixed `alpha` matters more
`RESOURCES.md:358`; Lau's CIFAR-10 test shows adaptives (Adam/RMSProp)
often beat hand-tuned schedules with less tuning `RESOURCES.md:184-194`;
step decay needs manual thresholds and is tedious `RESOURCES.md:191`.
Notebook Q8: time-based schedule reducing eta, distinct from momentum.
**Why not:** A says increase `alpha` over time; B says decay never
helps and forbids combining with adaptives; C claims decay and Adam
are mutually exclusive — they compose.
**Blog fix:** `blog.md:32` vague "like momentum" misses the
time-vs-gradient distinction.

### Q20. LR decay — formulas (details) → **A**
**Why A:** Inverse-time `alpha=alpha0/(1+decay_rate·epoch)` (Keras
`decay` per iteration) `RESOURCES.md:190,352,468`; step
`alpha0·drop^{floor(epoch/epochs_drop)}` `RESOURCES.md:191,470`;
exponential `alpha0·e^{−k·t}` or `0.95^{epoch}` `RESOURCES.md:192-193,
471`; inverse-time drops fast early, then slowly.
Notebook Q8 family: power/exponential/piecewise/1cycle are all
time-based.
**Why not:** B says only inverse-time exists and step/exponential are
momentum/RMSProp; C says `alpha=alpha0·(1+beta·t)` grows; D equates
decay to RMSProp's `s_dW` update `RESOURCES.md:156`.
**Blog fix:** `blog.md:32` called decay "like momentum" vague — now
three families with formulas.

---

### Q21. Feature scaling — both valid vs gamma·X+beta → **A**
**Why A:** Min-max `x'=(x−min)/(max−min)` → [0,1] `RESOURCES.md:40,454`
and standardization `x'=(x−mean)/std` → mean 0 var 1
`RESOURCES.md:42-43,453` are the two legitimate scaling options.
`gamma·X+beta` is batch norm's learnable affine
`RESOURCES.md:73,369` not input scaling. Purpose is to round the
elongated bowl so GD points toward the minimum `RESOURCES.md:64-68`,
not to prevent exploding gradients. Notebook Q1 correct is Both A
and B.
**Why not:** B claims only `gamma·X+beta` is scaling and that min-max
never used on inputs; C claims min-max equals `gamma·X+beta` and only
standardization is valid; D equates input scaling to per-batch BN
`gamma·z_norm+beta` `RESOURCES.md:71-76`.
**Econ bridge:** Like choosing between [0,1] share and z-score for
`income` vs `schooling` — both rescale, one bounds, one centers.
**Blog fix:** `blog.md:12-13` "normalization ((X*gamma)+beta)" →
replace with min-max formula and move gamma/beta to batch norm.

### Q22. Batch norm — every mini-batch, not interval schedule → **C**
**Why C:** BN normalizes `z` at every training step on the current
mini-batch: `mu, sigma2` from that batch, `z_norm=(z−mu)/
sqrt(sigma2+eps)`, `z_tilde=gamma·z_norm+beta` before `g()`
`RESOURCES.md:71-76,369-387,472`; `gamma, beta` updated with every
step like weights `RESOURCES.md:387`; `b` dropped because it cancels.
Notebook Q3 correct is every step per mini-batch
`source:zbqd84whl6a4kbcc1y9x`.
**Why not:** A is LR decay at fixed intervals `RESOURCES.md:190-191`
— the `blog.md:17` error; B restricts BN to a single pre-training
normalization of `X` only `RESOURCES.md:365` says hidden layers are
the point; D normalizes bias `b` at intervals, not activations.
**Blog fix:** `blog.md:17` "adjust gradient/learning rate only at
certain intervals" → fix to every mini-batch stable gradients and
higher safe learning rates.

### Q23. RMSProp — second moment only, not mean+variance → **B**
**Why B:** RMSProp keeps `s=beta·s+(1−beta)·dW²` element-wise, steps
`W−=alpha·dW/sqrt(s+eps)` `RESOURCES.md:156,322,465` — the EWMA of
squared gradients, i.e. the 2nd uncentered moment only
`source:zbqd84whl6a4kbcc1y9x` Q6. Sign is discarded; magnitude history
sets per-parameter scale `RESOURCES.md:158-159`.
**Why not:** A tracks signed mean `dW` without square — the 1st moment
`blog.md:27` error; C tracks both mean and variance together (1st+2nd)
as the blog claimed; D is min-max range, no averaging.
**Econ bridge:** Like GARCH `sigma_t^2 = omega + alpha·eps_{t-1}^2 +
beta·sigma_{t-1}^2` — decayed squares set next-step volatility.
**Blog fix:** `blog.md:27` "by its mean and variance, i.e. 1st and 2nd
moments" → fix to squared gradients only, second moment.

### Q24. LR decay — time-based schedule, not gradient-based → **D**
**Why D:** LR decay is a clock/epoch schedule shrinking global `alpha`:
inverse-time `alpha=alpha0/(1+decay·epoch)` `RESOURCES.md:190,352`,
exponential `alpha0·e^{−k·t}` `RESOURCES.md:192`, staircase
`alpha0·drop^{floor(epoch/p)}` `RESOURCES.md:191`; it is time-driven.
Momentum/RMSProp/Adam are gradient-driven (past `dW` / `dW²`)
`RESOURCES.md:107-111`. Notebook Q8 `source:zbqd84whl6a4kbcc1y9x`
correct is time-based schedule, not gradient-based, distinct from
momentum.
**Why not:** A is momentum's velocity `v=beta·v+(1−beta)·dW`
`RESOURCES.md:107`; B is RMSProp's per-parameter division by
`sqrt(s+eps)` `RESOURCES.md:156`; C equates decay to RMSProp's `s`
update.
**Blog fix:** `blog.md:32` "similar to momentum idea" → fix to
time-based scalar eta schedule (power/exponential/piecewise/1cycle)
vs gradient-based adaptivity.

---

#### Bridge to your blog rewrite (you will rewrite yourself)
* **Feature scaling:** keep standardization formula, fix min-max to
  `(x−min)/(max−min)`, drop `gamma,beta` there, state purpose = rounding
  elongated bowl / narrow valley, not exploding gradients
  `RESOURCES.md:64-68`; add Both-valid point from notebook Q1 →
  see Q21.
* **Batch norm vs mini-batch:** split completely — BN normalizes
  activations per batch every step `RESOURCES.md:71-76,472`; mini-batch
  is data splitting for updates `RESOURCES.md:234` → see Q22 vs Q8.
* **Momentum:** decay but accumulation building velocity in consistent
  direction `RESOURCES.md:308` → see Q11.
* **RMSProp:** second moment only (squared grads) `RESOURCES.md:156`
  → see Q23.
* **Adam:** both moments + two bias corrections `RESOURCES.md:173`
  → see Q16–Q17.
* **LR decay:** time-based schedule `RESOURCES.md:190-192` distinct
  from gradient-based momentum/RMSProp → see Q24.
* **Pros/cons:** every technique needs both — use Q1–Q24 trade-offs.
* **Econ bridges to keep:** income/schooling dollars, CPI base-year
  (train-only stats), GARCH volatility (RMSProp), census vs survey
  (mini-batch).

#### Verification
* No verbatim transcript >1 sentence; all summaries in own words per
  `RESOURCES.md` header.
* Every formula checked to `RESOURCES.md:105-112` and video summaries
  `RESOURCES.md:212-359` and notebook `source:zbqd84whl6a4kbcc1y9x`.
* Options length-balanced ±15% and equally detailed; correct positions
  6× each letter; high→low within each block.
* Blog line citations: `blog.md:12-13,17,20,22-25,27,30,32`.
