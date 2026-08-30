# Optimization — Answer Key & Explanations (20Q)

> Separate file — keep folded while you attempt `quiz_optimization_mcq.md`.
> Grounding: `RESOURCES.md` summaries + notebook
> `9e3bcac0-163a-465a-8167-63a63ad00dc7` (Gemini 2.5 grounded answers,
> captured 2026-08-30, session `a3706952`).

**Quick key:** 1C · 2B · 3D · 4A · 5C · 6B · 7D · 8A · 9C · 10B · 11D · 12A · 13C · 14B · 15D · 16A · 17C · 18B · 19D · 20A
*(Correct positions randomized ≈ 5× each letter: A=5, B=5, C=5, D=5.
Stems are high → low within each topic block — see header in MCQ file.)*

**Scoring (undergrad):** 18–20 excellent · 15–17 solid (review missed
topics) · 12–14 gaps — re-read `RESOURCES.md:448-478` Quiz Hooks ·
<12 redo with notebook.

---

### Q1. Feature scaling — why scale? → **C**
**Why C:** Different ranges stretch cost contours; weights live on
different scales so GD zig-zags and needs tiny `alpha`
`RESOURCES.md:64-68,223`. Scaling to zero-mean unit-variance rounds
the bowl so steps point toward the minimum.
**Why not:** A claims GD is scale-invariant like trees
`RESOURCES.md:57-58` (trees split on thresholds, GD uses geometry);
B confuses optimization with heteroskedasticity diagnostics; D
inverts the effect.
**Econ bridge:** `income` in dollars vs `schooling` in years — like
PPP vs local-currency GDP on one plot without rescaling.

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
— that's batch norm, not min-max.

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
regularization `RESOURCES.md:397-399`.
**Why not:** B is learning-rate decay, not BN — the exact confusion in
`blog.md:17`. C claims BN deletes `gamma, beta` to shrink the model
(opposite); D claims BN always hurts wall-clock time, ignoring faster
convergence offsetting per-epoch cost.

### Q5. Batch norm — core mechanics → **C**
**Why C:** Training: `mu, sigma2` per batch → `z_norm=(z−mu)/
sqrt(sigma2+eps)` → `z_tilde=gamma·z_norm+beta` → `a=g(z_tilde)`; bias
`b` cancels under mean subtraction so `beta` replaces it
`RESOURCES.md:71-76,369-387,473`; `gamma, beta` shape `(n^{[l]},1)`
`RESOURCES.md:386`.
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
`RESOURCES.md:234-242,458`.
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
**Why not:** A tracks mean `dW` without squaring — the `blog.md:27`
error claiming 1st+2nd moments; C is Adagrad's forever-growing sum; D
is batch norm's affine step.
**Blog fix:** `blog.md:27` said RMSProp uses “mean and variance (1st
and 2nd moments)” — only 2nd.
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
`m, v` start at 0 `RESOURCES.md:174`.
**Why not:** B uses only second moment; C is inverse-time LR decay;
D is batch norm `gamma·z_norm+beta` — name collision only.
**Blog fix:** `blog.md:30` said “RMS+momentum with bias correction” —
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
**Why not:** A says increase `alpha` over time; B says decay never
helps and forbids combining with adaptives; C claims decay and Adam
are mutually exclusive — they compose.

### Q20. LR decay — formulas (details) → **A**
**Why A:** Inverse-time `alpha=alpha0/(1+decay_rate·epoch)` (Keras
`decay` per iteration) `RESOURCES.md:190,352,468`; step
`alpha0·drop^{floor(epoch/epochs_drop)}` `RESOURCES.md:191,470`;
exponential `alpha0·e^{−k·t}` or `0.95^{epoch}` `RESOURCES.md:192-193,
471`; inverse-time drops fast early, then slowly.
**Why not:** B says only inverse-time exists and step/exponential are
momentum/RMSProp; C says `alpha=alpha0·(1+beta·t)` grows; D equates
decay to RMSProp's `s_dW` update `RESOURCES.md:156`.
**Blog fix:** `blog.md:32` called decay “like momentum” vague — now
three families with formulas.

---

#### Bridge to your blog rewrite (you will rewrite yourself)
* **Feature scaling:** keep standardization formula, fix min-max to
  `(x−min)/(max−min)`, drop `gamma,beta` there, state purpose = rounding
  contours + equalizing influence + stable `alpha`.
* **Batch norm vs mini-batch:** split completely — BN normalizes
  activations per batch `RESOURCES.md:72`; mini-batch is data splitting
  for updates `RESOURCES.md:234`.
* **RMSProp:** second moment only (squared grads); **Adam:** both
  moments + two bias corrections `RESOURCES.md:173`.
* **LR decay:** list 3 families with formulas and when to prefer
  adaptive `RESOURCES.md:189-193`.
* **Pros/cons:** every technique needs both — use Q1–Q20 trade-offs.

#### Verification
* No verbatim transcript >1 sentence; all summaries in own words per
  `RESOURCES.md` header.
* Every formula checked to `RESOURCES.md:105-112` and video summaries
  `RESOURCES.md:212-359`.
