# RNNs Resources

Ingested: 2026-10-12 · Source: [intranet project 2337 — RNNs](https://intranet-dlh.hbtn.io/projects/2337) (Master, weight 4)

18 "Read or watch" resources + 4 "Definitions to skim", all reached through the
authenticated rltoken links. Label/destination mismatches are flagged in the
bullets — the intranet labels do not always describe what the link actually is.

---

## MIT 6.S191 (2019): Recurrent Neural Networks

URL: https://www.youtube.com/watch?v=_h66BW-xNgk · Date: 2019-02-08 (YouTube upload) · Status: summary

Alexander Amini builds up to the RNN by showing why feed-forward nets fail on
sequences (fixed-length input, bag-of-words loses order, fixed windows can't
reach far context), then defines the recurrent cell as a state update that
consumes the current input and the previous hidden state, training with
backpropagation through time, and finally walking through LSTM's forget/update/
output operations as the remedy for vanishing gradients. The lecture closes with
applications — music generation, sentiment classification, and machine
translation with attention.

- Vanilla RNN cell: hidden state is a function of the current input and previous
  state, `a_t = act(W_xh @ x_t + W_hh @ h_{t-1} + b)`, with the output a separate
  projection `y_hat_t = W_hy @ h_t + b_y`. Both weight matrices are reused at
  every step (weight sharing).
- Unrolling: the loop drawn as a cycle is the same cell copied T times in a
  chain, `h_t = f(h_{t-1}, x_t)`; total loss is the sum of per-time-step losses.
- BPTT: gradients are computed at each time step then propagated back along the
  chain to `h_0`; each backward step multiplies by `W_hh` and the activation's
  derivative.
- Exploding vs vanishing: if those factors are > 1 the gradient blows up (fix
  with gradient clipping); if < 1 it decays toward zero, so early time steps
  barely train. Mitigations named: ReLU (derivative ≈ 1 for x > 0, versus < 1
  almost everywhere for tanh and sigmoid) and identity-style initialization.
- LSTM operations: forget gate `f_t = σ(W_f·[h_{t-1}, x_t] + b_f)` decides what to
  drop from `c_{t-1}`; input path `i_t = σ(...)` scales a tanh candidate
  `g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)`; cell update
  `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`; output `h_t = o_t ⊙ tanh(c_t)` with
  `o_t = σ(W_o·[h_{t-1}, x_t] + b_o)`.
- Why LSTM fixes the gradient: the `c` chain passes backward through only
  elementwise multiply and add — no matrix multiply per step — so the gradient
  path stays uninterrupted.

## Recurrent Neural Networks (RNNs), Clearly Explained!!!

URL: https://www.youtube.com/watch?v=AsNTP8Kwu80 · Date: 2022-07-11 (YouTube upload) · Status: summary

Josh Starmer motivates the RNN with a toy "StatLand" stock-price predictor,
feeding two or three days of scaled prices (low = 0, medium = 0.5, high = 1)
through a small network with a feedback loop to show that both yesterday's and
today's values influence tomorrow's prediction. He then unrolls the loop into
copies of one network to make the flow obvious, emphasizes that weights and
biases are shared no matter how far you unroll, and closes on why vanilla RNNs
are rarely used: the exploding and vanishing gradient problem.

- The cell: input and activation output `y_1 = act(x_1·W_1 + b_1)`; the key
  junction is a summation adding the previous step's `y_1·W_2` (the loop) to
  today's `x_2·W_1`, so both days reach the final output.
- Unrolling replaces the loop with T copies of the same cell rewired in a chain;
  the last copy's output is the prediction, and every earlier copy's output is
  discarded (or used as a per-step loss).
- Weight sharing: `W_1`, `b_1`, `W_2` are the same across all copies, so the
  parameter count never grows with sequence length.
- Exploding gradient: the first input's contribution is multiplied by `W_2` once
  per unroll, `x_1 · W_2^T`; with `W_2 = 2` and T = 50 this is astronomically
  large, forcing huge gradient-descent steps that bounce around instead of
  converging.
- Vanishing gradient is the same product with `W_2 < 1` (e.g. 0.5):
  `x_1 · 0.5^50` ≈ 0, so updates become tiny and early time steps never learn —
  the motivation for LSTMs.

## Introduction to Recurrent Neural Networks

URL: https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/ · Date: 2018-10-03 (published) · Status: summary

GeeksforGeeks' broad RNN primer covers the two defining parts of a recurrent
network — the recurrent neuron that carries a hidden state, and the unfolding
trick that turns the loop into a chain of layers — then walks through the forward
equations, backpropagation through time, the four input/output topologies, and
the family of variants (vanilla, bidirectional, LSTM, GRU). It closes with a
runnable Keras character-level text generator built on `SimpleRNN`. Intranet
label was "Introduction to RNNs"; the page's real title is the longer one shown
here, and it is a living page (last modified 2026-05-06), so treat the date as
the original publication.

- Core forward math, given twice in two notational styles:
  `h_t = σ(U·X_t + W·h_{t-1} + B)` with `Y = O(V·h + C)`, and
  `h_t = tanh(W_hh·h_{t-1} + W_xh·x_t)` with `y_t = W_hy·h_t` — both describe the
  same shared-weight recurrent cell.
- BPTT is stated as a sum over time:
  `∂L/∂W = ∂L/∂h_3 · Σ_{k=1..3} (∂h_3/∂h_k)·(∂h_k/∂W)`, i.e. every intermediate
  hidden state contributes a term.
- The derivative splits into an "explicit" part (how the state uses W directly)
  and an "implicit" part (through the previous hidden state):
  `∂h_3/∂W = ∂h_3⁺/∂W + (∂h_3/∂h_2)(∂h_2⁺/∂W)`.
- Notation pitfall: the same equations appear once with `U, W, V, B, C` and again
  with `W_hh, W_xh, W_hy` — the page never reconciles the two, so don't mix them
  in one derivation.
- Four topologies by input/output count: one-to-one, one-to-many (image
  captioning), many-to-one (sentiment), many-to-many (translation).
- Variants: vanilla (vanishing-gradient limited), bidirectional, LSTM
  (input/forget/output gates), GRU (fuses input+forget into one update gate, no
  cell state).

## Illustrated Guide to Recurrent Neural Networks: Understanding the Intuition

URL: https://www.youtube.com/watch?v=LHXXI4-IEns · Date: 2018-08-26 (YouTube upload) · Status: summary

Michael (learnedvector) builds intuition without math, using a moving-ball
snapshot thought experiment and the alphabet-recall test to define "sequential
memory," then shows the hidden state acting as a highway carrying past inputs
forward. He walks a chatbot intent-classification example word by word to show
that the final output already encodes the whole sentence, stresses that a forward
pass is just a for-loop over tokens, and explains short-term memory as the
vanishing gradient shrinking through time.

- Sequential memory = recognizing and storing the order of a series of events;
  the hidden state `h_t` is the network's running summary of every input seen up
  to step `t`.
- Forward pass in code: initialize `h_0`, then `for word in sentence: out, h_t =
  rnn(word, h_{t-1})`; the last `out` goes to a dense layer for the prediction.
  Same weights at every iteration.
- The recurrent loop is literally a highway from `h_{t-1}` into `h_t`, the only
  path by which previous information can affect later ones (a plain feed-forward
  net has none).
- Short-term memory: in the "what time is it" example the hidden state at the
  last step has all but lost "what" and "time", so the classifier guesses "is
  it" — information from early tokens never reaches the output.
- Cause: backprop computes each node's gradient from the layer before, so
  gradients shrink exponentially with depth; in an RNN each time step is treated
  as a layer (BPTT) and early steps get vanishingly small updates.
- Fix directions: LSTM and GRU add learned gates that add or remove information
  from the hidden state; slower per step but they train on long sequences.

## Illustrated Guide to LSTM's and GRU's: A step by step explanation

URL: https://www.youtube.com/watch?v=8HyCNIVRbSU · Date: 2018-09-19 (YouTube upload) · Status: summary

Same author, now inside the LSTM cell: he contrasts a vanilla RNN (input and
previous hidden state combined, passed through tanh, bounding values to (-1, 1)
so they can't explode) with the LSTM's cell state plus three sigmoid gates, then
explains the GRU's simplification. He uses a product-review thought experiment to
motivate selective memory and gives pseudocode for both architectures.

- RNN cell: `h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b)` — the tanh keeps
  activations bounded in (-1, 1) so repeated multiplications don't blow up.
- LSTM's cell state `c_t` is a separate transport highway running along the whole
  unrolled chain, so information from step 1 reaches the last step without being
  rewritten at each step.
- Gates are small neural nets with sigmoid activation: output near 0 erases the
  value, near 1 passes it through unchanged — that's how the cell learns what to
  keep or drop.
- Forget / input / output sequence: `f_t = σ(W_f·[h_{t-1}, x_t] + b_f)`; candidate
  `g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)` scaled by `i_t = σ(W_i·[h_{t-1}, x_t] + b_i)`;
  `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`; `h_t = o_t ⊙ tanh(c_t)` with
  `o_t = σ(W_o·[h_{t-1}, x_t] + b_o)`.
- Pseudocode control flow: concatenate `h_{t-1}` and `x_t` into `combined`, feed
  it to the forget layer, the tanh candidate layer, and the input layer, compute
  `c_t` from those three vectors plus the previous `c_{t-1}`, then pointwise-
  multiply the output gate with the new cell state to get `h_t` — again, just a
  for-loop.
- GRU: no cell state at all (the hidden state carries the information), only two
  gates — update (plays the combined role of LSTM's forget + input) and reset
  (how much past state feeds the candidate) — so fewer tensor ops and a bit
  faster to train than LSTM.

## Long Short-Term Memory (LSTM), Clearly Explained

URL: https://www.youtube.com/watch?v=YCzL96nL7j0 · Date: 2022-11-07 (YouTube upload) · Status: summary

This lesson rebuilds the LSTM cell gate by gate, running explicit numbers through
one cell so each part's role is unambiguous. The core idea is a split between two
memory paths: a cell state carrying long-term memory along the top of the cell
and a hidden state carrying short-term memory, so information no longer has to
survive the feedback loop through weights. Every gate is a sigmoid that outputs a
fraction in [0, 1], while tanh supplies candidate values in [-1, 1], and the
worked stock-price example shows why the unrolled cell can hold a day-1 value
that a vanilla RNN would have lost.

- The cell state (long-term memory) is modified only by multiplication and
  addition — no weights or biases sit directly on that path, so gradients travel
  through it unattenuated, which is why unrolling can cover much longer sequences
  than a vanilla RNN.
- Forget gate: `f_t = σ(W_f·h_{t-1} + U_f·x_t + b_f)` multiplies the previous cell
  state `c_{t-1}`, setting the percentage of long-term memory kept. In the video,
  `c_{t-1}=2, h_{t-1}=1, x_t=1` gives pre-activation 5.95, `f_t=0.997`, and
  `2 × 0.997 = 1.99` — a tiny reduction; a large negative input drives `f_t` to 0
  and wipes the memory.
- Candidate: `c̃_t = tanh(W_c·h_{t-1} + U_c·x_t + b_c)` (0.97 in the run) is a
  potential long-term memory built from the previous hidden state and the current
  input; a strongly negative input pushes it to −1.
- Input gate: `i_t = σ(W_i·h_{t-1} + U_i·x_t + b_i)` scales the candidate before it
  is stored. With sigmoid output 1.0 the whole candidate is admitted, giving
  `c_t = 1.99 + 0.97 = 2.96`; with a very negative input the gate is 0 and nothing
  is added.
- Output gate: feed the new cell state through tanh for a potential short-term
  memory (2.96 → 0.99), then multiply by `o_t = σ(W_o·h_{t-1} + U_o·x_t + b_o)`
  (0.99) to produce `h_t = o_t ⊙ tanh(c_t) = 0.98`. That `h_t` is both the new
  short-term memory and the cell's output — hence the gate's name.
- Weight sharing across time is what makes variable-length sequences possible:
  the unrolled cell reuses identical weights at every step, so two companies
  whose day-1 values differ but whose days 2–4 match still produce different
  day-5 predictions.

## Gated Recurrent Unit Networks

URL: https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/ · Date: 2019-07-09 (published) · Status: summary

This page presents the GRU as the cheaper sibling of the LSTM: same goal of
tracking long-range dependencies, but only two gates and no separate cell state,
so it trains faster with fewer parameters. It gives the four GRU equations
explicitly, a side-by-side GRU/LSTM comparison table, and a Keras time-series
forecasting walkthrough on 8,000 days of temperature data with sliding windows.
The intranet label adds "(GRU) Explained"; the page itself is titled simply
"Gated Recurrent Unit Networks" and is a living page (last modified 2026-06-11).

- Reset gate: `r_t = σ(W_r · [h_{t−1}, x_t] + b_r)` — decides how much of the past
  hidden state feeds the candidate.
- Update gate: `z_t = σ(W_z · [h_{t−1}, x_t] + b_z)` — decides the retain-vs-
  refresh balance.
- Candidate (proposed) state: `h′_t = tanh(W_h · [r_t ⊙ h_{t−1}, x_t] + b_h)`, with
  the reset gate applied multiplicatively to the memory before it is mixed with
  the current input.
- Final state is a convex blend: `h_t = (1 − z_t)·h_{t−1} + z_t·h′_t`. This
  convention matches colah's LSTM post, where `z` multiplies the *new* candidate;
  some papers use the mirrored convention, so state which one you use.
- GRU vs LSTM: 2 gates vs 3, hidden state only vs separate cell state, lower
  parameter count and faster training, comparable accuracy on most tasks.
- Keras shape fingerprints: `GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1))`,
  then `GRU(units=50)`, `Dense(units=1)`, `Adam(learning_rate=0.001)`,
  `loss='mean_squared_error'`; input must be reshaped to 3-D
  `(samples, time_steps, features)`.

## Recurrent Neural Networks : Introduction for Beginners

URL: https://www.analyticsvidhya.com/blog/2021/06/recurrent-neural-networks-introduction-for-beginners/ · Date: 2021-06-13 (published) · Status: summary

A short blogathon post that introduces the recurrent cell for newcomers: why
feedforward nets fail at next-word prediction, what unfolding means, and how
parameter sharing shrinks the parameter count. It then summarises forward
propagation, a four-step backpropagation recipe, the vanishing/exploding gradient
issue, the four architecture types, and bidirectional RNNs. Intranet label was
"RNNs Tutorial, Parts 1"; the real page title is "Recurrent Neural Networks :
Introduction for Beginners". Full body was readable; the "Login to continue
reading" line is a footer prompt, not a truncation.

- Forward state: `h_t = f(h_{t-1}, x_t)`, expanded as
  `h_t = tanh(w_hh·h_{t−1} + w_xh·x_t)`; output `O_t = W_hy·h_t`.
- Parameter roles are spelled out explicitly: `U` maps input `x_t` to state `s_t`,
  `W` maps previous state to current state, `V` maps state to output — and the
  same three are reused at every unfolded step.
- Loss is cross-entropy `L = −ln(p_c)` paired with softmax; the worked number is a
  95%-confident correct prediction giving ≈ 0.051.
- Backprop recipe: unroll all time steps → compute per-step gradients w.r.t. each
  weight → **because the weight is identical across steps, sum the time-step
  gradients** → update recurrent and dense weights.
- Vanishing vs exploding trigger: gradients below 1 shrink exponentially through
  repeated chain-rule products; gradients above 1 blow up.
- Bi-directional RNN runs a forward and a backward hidden layer over the reversed
  sequence and concatenates per-timestep outputs — needed for tasks where the
  whole sequence is available.

## Fundamentals of RNN forward Propagation in Deep Learning

URL: https://www.analyticsvidhya.com/blog/2017/12/introduction-to-recurrent-neural-networks/ · Date: 2017-12-07 (published) · Status: summary

This is the longest of the article resources: it argues from sequence-sensitive
problems (language, time series, genomes) to the recurrent neuron, then traces a
full forward pass **by hand in Excel** using a four-letter vocabulary before
giving a Keras implementation. It also covers BPTT at a conceptual level and
explains why exploding gradients are easier to handle than vanishing ones.
Label/title mismatch to note: the intranet calls this "RNNs Tutorial, Parts 2",
but the real page title is "Fundamentals of RNN forward Propagation in Deep
Learning" and its scope is forward propagation (the URL slug still reads
`introduction-to-recurrent-neural-networks`).

- Task motivation: RNNs map inputs to outputs of varying type and length —
  sentiment classification (variable in, fixed out), image captioning (fixed in,
  variable out), translation (variable in and out).
- Why weights are shared: stacking per-layer weights makes each hidden layer
  behave independently; forcing one shared `W`/`b` lets the separate layers be
  collapsed into a single recurrent neuron applied at every time step.
- State recursion (worked on "hello" with vocabulary {h,e,l,o}, one-hot inputs, a
  3×4 `W_xh`, 1×1 `W_hh`): `h_t = tanh(W_xh·x_t + W_hh·h_{t−1} + b)`, then
  `y_t = W_hy·h_t` and `softmax(y_t)` over the vocabulary.
- Seven-step forward loop per time step: supply `x_t`, compute `h_t` from `x_t` and
  `h_{t−1}`, shift `h_t → h_{t−1}`, repeat to the end, then compute `y_t`, compare
  with the target, and backpropagate.
- BPTT is presented as plain backprop on the unrolled graph: per-step
  cross-entropy `E_t = −ȳ_t log(y_t)`, total `E = −Σ_t ȳ_t log(y_t)` (whole
  sequence = one training example), gradients per step summed before the update.
  Math-heavy detail is deferred to a promised follow-up.
- Vanishing gradient illustrated on "The man who ate my pizza has purple hair":
  the chain drives gradients to zero if any factor is near 0; clipping fixes
  exploding gradients, whereas LSTM/GRU gating is the remedy for vanishing.

## Understanding LSTM Networks

URL: https://colah.github.io/posts/2015-08-Understanding-LSTMs/ · Date: 2015-08-27 (published) · Status: summary

Christopher Olah's canonical illustrated LSTM essay. It motivates recurrence with
the "information persists" argument, shows that long-term dependencies fail in
practice despite being learnable in theory, then walks the LSTM diagram gate by
gate and finishes with variants. Note the mismatch: the intranet labels this
"RNNs Tutorial, Parts 3", but the destination is an LSTM post, and the intranet
itself warns that its final GRU equation is wrong — treat the GRU variant
paragraph with suspicion and use a primary source for GRU.

- Notation is declared explicitly: lines are full vectors, pink circles are
  pointwise ops (add/multiply), yellow boxes are learned layers, merging lines =
  concatenation forking = copy. This is the notation the DLH tasks reuse.
- Vanilla RNN cell: `h_t = f(W_hh·h_{t−1} + W_xh·x_t)` with the repeating module
  being a single `tanh` layer; unrolling it into a chain shows why sequence
  problems demand recurrence.
- LSTM repeating module has four interacting layers; the horizontal top line is
  the **cell state** `C_t`, described as a near-conveyor-belt carrying
  information with only minor linear interactions — the additive path that keeps
  gradients alive.
- Gates are sigmoid layers producing values in [0,1] (0 = block fully, 1 = pass
  fully) multiplied pointwise into the flow: forget gate `f_t`, input gate `i_t`
  with tanh candidate `C̃_t`, output gate `o_t` giving `h_t = o_t ⊙ tanh(C_t)`.
- Cell update is additive, not multiplicative: `C_t = f_t ⊙ C_{t−1} + i_t ⊙ C̃_t` —
  adding rather than multiplying is why long-range gradients survive.
- Variants: peephole connections (Gers & Schmidhuber 2000), coupled forget+input
  gates, and the GRU (Cho et al. 2014), which merges forget and input into one
  update gate and fuses cell state with hidden state; Greff et al. 2015 find
  variants roughly equivalent.
- **Intranet warning applies here**: the GRU equation in this post's variants
  section is flagged as incorrect — prefer the Cho et al. 2014 form
  (`h_t = (1−z_t)·h_{t−1} + z_t·h̃_t` with
  `h̃_t = tanh(W[r_t⊙h_{t−1}, x_t] + b)`, matching the GeeksforGeeks GRU page)
  when writing GRU code.

## Bidirectional RNN Indepth Intuition - Deep Learning Tutorial

URL: https://www.youtube.com/watch?v=D-a6dwXzJ6s · Date: 2020-05-30 (YouTube upload) · Status: failed-transcript

Krish Naik, 7:54. No transcript available: `youtube-transcript-api` reports
`NoTranscriptFound`, and the documented `yt-dlp --write-auto-subs` fallback
reports that automatic captions are missing for this video (a PO token would be
required), so no captions exist to pull. Covered by the "Deep RNNs & Bi RNNs"
summary below and by the BRNN definition; watch manually if you want the
narrated version.

## Training RNNs | Loss & BPTT | Machine Learning for Engineering & Science Applications

URL: https://www.youtube.com/watch?v=RrB605Mbpic · Date: 2019-05-06 (YouTube upload) · Status: summary

This lecture covers the two things that make RNN training distinct from CNN/ANN
training: how the loss is formed when an output is read at many (or all) time
steps, and how backprop must then travel backward through those same steps. It
works through the derivative calculus for a shared-parameter recurrent cell,
keeping the output activation linear and the loss quadratic so the chain rule
stays readable, and stresses that in practice frameworks resolve all of it
through graph-based autodiff. The running thread is that the same weight matrix
appearing at every step is exactly what creates the recursion — and, as the
sequence lengthens, that recursion is where the vanishing/exploding-gradient
problem enters.

- Loss accumulates over time: with outputs read at every step there is one local
  loss `L_t` per step against ground truth `y_t`, and the objective is the sum
  `L = Σ_{t=1}^{T} L_t` — not a single loss at the end as with a typical ANN/CNN.
- Motivating example: given daily temperatures for one city, predict the next
  day's temperature at each step (`ŷ_1 … ŷ_T` against `y_1 … y_T`); where outputs
  are read is task-dependent, and intermediate outputs are optional.
- Per-step loss form is picked by task — least squares for numeric regression,
  cross-entropy for classification (will it rain, yes/no).
- Forward pass with three shared matrices: `h_t = g(W·h_{t-1} + U·x_t)`,
  `ŷ_t = g*(V·h_t)`; `W`, `U`, `V` do not change with time, which both cuts the
  parameter count and is the source of every subtlety below.
- Easy case: `∂L_t/∂V = (y_t − ŷ_t)·h_tᵀ`. Note the ordering — the gradient must be
  a matrix, so the error and hidden vectors form an outer product.
- Hard case (the point of "through time"):
  `∂h_t/∂W = g'(z_t)·h_{t-1} + W·∂h_{t-1}/∂W`, where `z_t = W·h_{t-1} + U·x_t`. The
  second term is nonzero only because the same `W` appeared at step `t−1`, so the
  derivative cannot be evaluated locally and must be walked back to the start of
  the sequence. The input path mirrors it:
  `∂h_t/∂U = g'(z_t)·(x_t + U·∂h_{t-1}/∂U)`.

## Recurrent Neural Networks Tutorial, Part 3 – Backpropagation Through Time and Vanishing Gradients

URL: https://dennybritz.com/posts/wildml/recurrent-neural-networks-tutorial-part-3/ · Date: 2015-10-08 (published) · Status: summary

Denny Britz's third tutorial part derives BPTT step by step with concrete
indices, gives naive Python/NumPy code for it, and then uses the same gradient
expression to explain why RNNs cannot learn long-range dependencies. The
vanishing-gradient section is the bridge that motivates LSTM and GRU. This is the
mathematical backbone of the `rnn` project's BPTT and cell tasks.

- Notation used throughout: `s_t = tanh(U·x_t + W·s_{t−1})` and
  `ŷ_t = softmax(V·s_t)` (output relabelled `ŷ`, not `o`, to match the
  literature); total loss `E = −Σ_t y_t log ŷ_t` summed over the whole sequence.
- Gradients sum over time for one example: `∂E/∂W = Σ_t ∂E_t/∂W`; for `V` the
  gradient is purely local, `∂E_3/∂V = (ŷ_3 − y_3) ⊗ s_3`, an outer product —
  hence the `np.outer` in code.
- For `W` you must re-apply the chain rule because `s_3` depends on `s_2` depends
  on `W`: `∂E_3/∂W = Σ_{k=0..3} ∂E_3/∂ŷ_3 · ∂ŷ_3/∂s_3 · ∂s_3/∂s_k · ∂s_k/∂W`.
- Key insight: BPTT is ordinary backprop on the unrolled RNN, with one difference
  from feedforward nets — gradients must be **summed** across time steps because
  `W` is shared. Delta form used:
  `δ_2^(3) = ∂E_3/∂z_2 = ∂E_3/∂s_3 · ∂s_3/∂s_2 · ∂s_2/∂z_2`.
- The code loop backprops with `delta_t = W.T.dot(delta_t) * (1 - s[t-1]**2)` —
  the `1 − s²` is the `tanh` derivative — and supports truncated BPTT via
  `bptt_truncate`, which the author calls standard practice for long sequences.
- Vanishing mechanism: `∂s_3/∂s_k` is itself a product of Jacobians whose 2-norm
  is bounded by 1 (by 1/4 for sigmoid); saturated tanh/sigmoid units have ~0
  derivative, so repeated multiplication shrinks the gradient exponentially.
  Exploding = the same product with large Jacobian entries; it is fixable by
  clipping at a threshold, vanishing is not.

## Deep RNNs & Bi RNNs | Machine Learning for Engineering & Science Applications

URL: https://www.youtube.com/watch?v=OLO6kyxwqnY · Date: 2019-05-06 (YouTube upload) · Status: summary

Two structural variants are covered here, both presented as small modifications of
the standard recurrent cell rather than new ideas. A deep RNN simply swaps each
recurrent layer for a deep network, turning the diagram into a grid with time
running horizontally and depth vertically, which introduces per-level weight
matrices and multiple gradient paths. A bidirectional RNN adds a second set of
hidden units that runs the sequence backwards, and the output combines forward and
backward states — motivated by tasks such as handwriting recognition and
whole-sentence translation where the label depends on context arriving later in
time.

- Depth direction vs time direction: horizontally the diagram is the sequence;
  vertically it is depth. What looks like a wide network is just one deep network
  unrolled, with the same structure and weights repeated.
- The only change from a plain RNN is that what arrives below a cell is no longer
  the input `x_t` but the cell's own activation from the level below:
  `h_t^(l) = tanh(W^l·h_{t-1}^(l) + U^l·h_t^(l-1))`.
- Each level owns its own pair of matrices `(W^l, U^l)` — that per-level indexing
  is the entire difference between deep and shallow RNN.
- Extra difficulty: gradient paths now multiply (down, back in time, across
  levels), so backprop gets more tangled — exactly what graph-based autodiff in
  TensorFlow handles for you.
- Bidirectional variant: keep `h_t = tanh(W_f·h_{t-1} + U_f·x_t + b_f)` and add a
  reverse copy `h_t^r = tanh(W_r·h_{t+1}^r + U_r·x_t + b_r)`, where each reverse
  step consumes the *next* input rather than the previous one.
- The output reads from both directions at once:
  `ŷ_t = g(V_f·h_t^f + V_r·h_t^r + b)`. In the lecture's simple example that comes
  to nine parameters — three forward cell, three reverse cell, the two output
  matrices, and the output bias.
- Why it matters: reading "SO?" left-to-right alone cannot distinguish T from F,
  but reading the word backward resolves it; likewise a sit/stand sensor at the
  first instant of a signal is only classifiable from what happens later.

## Deep Recurrent Networks

URL: https://cedar.buffalo.edu/~srihari/CSE676/10.5%20DeepRecurrent.pdf · Date: 2017-07-06 (course lecture) · Status: summary

An 8-slide CSE 676 lecture deck by Sargur Srihari (University at Buffalo)
asking a single question: is it worth putting depth inside an RNN? It starts by
decomposing a standard RNN into its three parameter blocks (input→hidden,
hidden→hidden, hidden→output), notes that each is only a *shallow* transformation
(one learned affine map plus a fixed nonlinearity, i.e. one layer of an MLP), and
then walks through three routes to adding depth — hierarchical grouping of the
recurrent state, a separate (deep) MLP per block, and skip connections. The deck's
payload is the trade-off: extra depth buys representational capacity but lengthens
the shortest path between time steps, which is bad for gradient flow, and skip
connections exist purely to buy back that path length.

- The three parameter blocks are: input→hidden state, previous hidden→next
  hidden, hidden→output; each corresponds to exactly one weight matrix, i.e. one
  shallow transformation.
- Route 1: split the hidden recurrent state into groups arranged hierarchically,
  with lower levels refining the raw input into something more useful higher up.
- Route 2: replace each of the three blocks with its own possibly-deep MLP —
  representational capacity argues for it, but shallower architectures are
  generally easier to optimize.
- Path-lengthening cost: with a single hidden layer in the state-to-state
  transition, the shortest path between variables at any two different time steps
  is *doubled* relative to an ordinary RNN.
- Route 3: introduce skip connections in the hidden-to-hidden path to mitigate
  that path-lengthening; this is the concrete mitigation drawn on the final slide.

## Training and Analyzing Deep Recurrent Neural Networks

URL: https://papers.nips.cc/paper_files/paper/2013/file/1ff8a7b5dc7a7d1f0ed65aaa29c04b1e-Paper.pdf · Date: 2013-12-02 (published) · Status: summary

NIPS 2013 paper (Advances in NIPS 26, pp. 190–198) by **Michiel Hermans and
Benjamin Schrauwen** (Ghent University, ELIS department). **Attribution
correction:** the intranet label pairs this link with Pascanu/Mikolov/Bengio's
clipping-threshold work, but the PDF's title page and metadata name Hermans &
Schrauwen, and the content matches their deep-recurrent-hierarchy work — no
spectral-radius analysis appears in it. It defines a Deep RNN as a stack of L
recurrent layers where each layer consumes the previous layer's hidden-state *time
series* as its input, argues this buys an explicit temporal hierarchy, and
demonstrates it on next-character prediction over a 1.4-billion-character Wikipedia
corpus with plain SGD, matching a multiplicative RNN that had needed Hessian-free
optimization.

- Architecture: `a^i(t) = tanh(W^i a^i(t-1) + Z^i ā^{i-1}(t))` for i > 1, and the
  same with the raw input `s(t)` for i = 1; biases are handled by augmenting
  vectors with a trailing 1. Each layer's fading memory sits one level higher in
  the time hierarchy than the one below.
- Two output schemes: **DRNN-1O** (only the top layer feeds the softmax) and
  **DRNN-AO** (softmax over the sum of all layers' outputs,
  `softmax(Σ_i U^i ā^i(t))`). All-layer outputs exist because BPTT error shrinks
  before reaching lower layers — so lower layers otherwise go untrained — and
  because zeroing a layer's output is a cheap ablation probe.
- Gradient handling is **normalization, not clipping**:
  `θ(j+1) = θ(j) − η₀(1 − j/T)·∇θ(j)/‖∇θ(j)‖`, a heuristic that avoids huge
  gradients near bifurcations. There is no clipping threshold in this paper.
- Setup: ~4.9M parameters each (RNN 2119 units; DRNN-1O 5×727; DRNN-AO 5×706), 95
  characters + unknown, metric BPC = −⟨log₂ p_c⟩, BPTT on 75 parallel 250-char
  sequences with the first 50 chars excluded from the backward pass, output
  weights zero-initialised, `η₀ = 0.5`.
- Results: **RNN 1.610, DRNN-AO 1.557, DRNN-1O 1.541** BPC, versus MRNN 1.55
  (Hessian-free, 5 days, 8 GPUs). Training took ~10 days per network on one
  GTX 680.
- Ablation: zeroing one layer's output hurts the top layer most and the bottom
  layer second; intermediate layers matter mainly as preprocessing for the top
  layer. Removing the bottom layer collapses generated text into runs of rare
  characters.
- Long-range dependency test: the network keeps closing-parenthesis probability
  elevated for the entire span between brackets (demonstrated over 296
  characters), behaving as a pseudo-stable attractor state rather than memorised
  phrase lengths; it cannot handle nesting. Generation failure rates:
  **DRNN-AO 12%, DRNN-1O 16%, RNN 28%**.

## Lecture 15: Exploding and Vanishing Gradients

URL: https://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L15%20Exploding%20and%20Vanishing%20Gradients.pdf · Date: 2017-03-28 (course lecture) · Status: summary

Roger Grosse's CSC321 lecture notes explaining why BPTT gradients explode or
vanish over long windows, first mechanically (the Jacobian is a product of
per-step derivatives) and then intuitively (an RNN is an iterated dynamical system
with attractors), before covering the fixes: gradient clipping, input reversal,
identity initialisation, and finally the LSTM. A recurring theme is that the
forward pass is protected by squashing nonlinearities while the backward pass is
purely linear, so nothing stops the derivatives from blowing up; and that the
exploding/vanishing boundary in the nonlinear case is not at w = 1 but depends on
where the hidden state currently sits.

- Mechanically: iterating `h(t) = z(t+1)w`, `z(t) = h(t)φ′(z(t))` gives
  `h(1) = w^{T-1}φ′(z(2))···φ′(z(T))h(T)`. With linear activations this is just
  `w^{T-1}`: **w = 1.1, T = 50 → 117.4**, while **w = 0.9, T = 50 → 0.00515**.
- Multivariate form: the Jacobian `∂h(T)/∂h(1) = ∏_t ∂h(t+1)/∂h(t)`, which for
  linear activations is `W^{T-1}` — it explodes if the **largest eigenvalue of W > 1**
  and vanishes if it is < 1.
- Intuition via iterated functions: expanding `h(4) = f(f(f(h(1),x(2)),x(3)),x(4))`
  makes the RNN an iteration of `f`. For the monotonic `f(x) = x² + 0.15` on
  [0,1], fixed points satisfy **f′(x) < 1 → sink/attractor** and **f′(x) > 1 →
  source/repeller**. Inside a basin of attraction the Jacobian is ≈ 0; on the
  basin boundary it is large.
- Loss-surface consequence: per-example losses develop **cliffs** (a huge step can
  fling you off the good region), though the cost averaged over 1000 examples is
  comparatively smooth.
- Gradient clipping: if `‖g‖ > η`, rescale to `g ← η·g/‖g‖`. This deliberately
  biases training — the result is no longer the true cost gradient — but keeps
  optimisation inside the narrow valley. **The notes leave η as a symbolic
  hyperparameter and give no specific threshold value.**
- Input reversal: for word-order-similar language pairs every dependency gap is
  about a sentence long; reversing the input brings the first word's gap down to a
  single time step.
- Identity initialisation: iterating the identity map is perfectly stable since
  the Jacobian is just I. The identity RNN (Le et al., 2015) uses ReLU units with
  recurrent weights set to the identity matrix — ReLU equals the identity on
  nonnegative activations.
- LSTM block: gates `i, f, o` via logistic, candidate `g` via tanh, with
  `c(t) = f(t)∘c(t−1) + i(t)∘g(t)` and `h(t) = o(t)∘tanh(c(t))`. Truth table:
  (input, forget) = (0,1) remembers, (1,1) adds, (0,0) erases, (1,0) overwrites.
- Why it works: the exercise is to show that with the forget gate on and
  input/output gates off, memory-cell gradients pass through **unmodulated** at
  every step — so LSTMs resist explosion and vanishing, though neither is
  mathematically impossible.

## Vanishing and Exploding Gradients Problems in Deep Learning

URL: https://www.geeksforgeeks.org/deep-learning/vanishing-and-exploding-gradients-problems-in-deep-learning/ · Date: 2023-12-13 (published) · Status: summary

A practical reference page defining both gradient pathologies, listing their causes
and standard cures, and ending with a reproducible experiment that trains two
identical 20-layer networks — one sigmoid, one ReLU — and plots the two loss
curves so the vanishing effect is visible. The intranet told you to stop at section
4, which is the last of the four fix techniques before the code; note that the
gradient-clipping entry there is a single sentence with no formula, so get
clipping's actual update rule elsewhere (see the Grosse lecture above). Real page
title differs from the intranet label by adding "in Deep Learning".

- Vanishing gradient:
  `∂L/∂w_i = (∂L/∂a_n)·(∂a_n/∂a_{n−1})·…·(∂a_1/∂w_i)` — a chain of small factors
  repeatedly multiplied, so early layers get near-zero updates and effectively
  stop learning.
- Exploding gradient: the mirror case,
  `Π_{i=1..n} ∂a_i/∂a_{i−1} → ∞` when the factors exceed 1; feeding that into
  the update `w_{t+1} = w_t − η·∂L/∂w_t` makes weight jumps enormous and the loss
  oscillate or diverge.
- Named causes: saturating activations (sigmoid, tanh) with tiny derivatives, poor
  weight initialization, depth multiplying many factors, and too-high learning
  rates or unscaled inputs.
- Four fixes in the order the page gives them: **1** Xavier or Kaiming
  initialization, **2** non-saturating activations (ReLU, Leaky ReLU, ELU/SELU),
  **3** batch normalization (zero mean, unit variance per layer), **4** gradient
  clipping — clipping is described only as clamping gradients to a maximum
  threshold, with no pseudocode or threshold value.
- Demonstration: `build_model(activation, layers=20)` with
  `Dense(32, activation=...)` repeated, a final `Dense(1, activation='sigmoid')`,
  `Adam(learning_rate=0.001)`, `loss='binary_crossentropy'`; sigmoid loss creeps
  down slowly, ReLU loss drops fast — the loss curve is the practical readout of
  gradient flow.
- RNN-specific framing: because the state is reused across time steps, vanishing
  gradients destroy long-term memory while exploding ones cause divergence, which
  is why LSTM/GRU gating plus clipping, initialization and normalization are the
  standard stability toolkit for sequence models.

---

## Definitions to Skim

One-line glossary entries, per the intranet's "Definitions to skim" section.

| Term | Definition | URL · Date |
|------|------------|------------|
| **RNN** | A neural network whose cell feeds its own output back as the next input, giving it memory of previous steps. | [Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) · 2026-10-04 (last edited) |
| **LSTM** | A gated RNN variant with a cell state and forget/input/output gates that keeps gradients from vanishing over long sequences. | [Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory) · 2026-08-10 (last edited) |
| **GRU** | A lighter gated RNN with only a reset and an update gate and no separate cell state. | [Gated recurrent unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit) · 2026-08-06 (last edited) |
| **BRNN** | A bidirectional RNN that runs a second cell backwards over the sequence and combines both directions. | [Bidirectional recurrent neural networks](https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks) · 2025-11-07 (last edited) |

---

## Ingestion Notes

- Every rltoken was resolved inside the authenticated browser; the intranet labels
  below were mismatched on 4 of 18 "Read or watch" items.
- Mismatches found: "RNNs Tutorial, Parts 2" → a forward-propagation walkthrough;
  "RNNs Tutorial, Parts 3" → colah's LSTM essay; "GRU Explained" → an article, not a
  video; "Training and Analyzing Deep Recurrent Neural Networks" → the
  Hermans & Schrauwen paper, not the Pascanu et al. one.
- The intranet's own note ("slight mistake in the last equation for the GRU cell")
  refers to the colah post's variants section; use the Cho et al. 2014 convention
  `h_t = (1 − z_t)·h_{t−1} + z_t·h̃_t`.
- One video has no captions at all (`D-a6dwXzJ6s`) — both documented retrieval
  paths fail, so it is marked `failed-transcript` rather than summarized.

---

## Quiz Hooks

- RNN — network with memory: the hidden state is a function of the current input and the previous hidden state, with weights shared across time.
- Unrolling — the recurrent loop drawn as a chain of T copies of the same cell; nothing is copied in the implementation.
- Weight sharing — `W_hh`, `W_xh`, `W_hy` are reused at every time step, so the parameter count does not grow with sequence length.
- BPTT — ordinary backprop on the unrolled graph, with the twist that gradients must be summed over time steps because the weights are shared.
- `∂h_t/∂W = g'(z_t)·h_{t-1} + W·∂h_{t-1}/∂W` — the recurrent term is why the derivative cannot be computed locally in time.
- Per-step loss — `L = Σ_t L_t`; one loss per time step when outputs are read everywhere, instead of a single final loss.
- Vanishing gradient — products of factors below 1 shrink exponentially with sequence length; early time steps stop learning.
- Exploding gradient — the same product with factors above 1; fixable by clipping (`g ← η·g/‖g‖`), vanishing is not.
- Spectral view — the Jacobian over T steps is `W^{T-1}`; it explodes when the largest eigenvalue of W exceeds 1.
- LSTM cell state `c_t` — a conveyor-belt path updated only by elementwise multiply and add, so gradients flow through it unattenuated.
- LSTM gates — forget, input, output: sigmoids in [0,1] that decide what to erase, what to write, and what to expose as `h_t`.
- LSTM update — `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`, `h_t = o_t ⊙ tanh(c_t)`; the add (not multiply) is the whole trick.
- GRU update — `h_t = (1 − z_t)·h_{t−1} + z_t·h̃_t`; two gates, no cell state, cheaper than LSTM.
- BRNN — a second cell running the sequence backwards, outputs concatenated; needed when the label depends on later context ("SO?" → T vs F).
- Deep RNN — stacking recurrent layers; each level gets its own `(W^l, U^l)`, and the shortest path between time steps gets longer.
- Truncated BPTT — cut the backward pass to a fixed window (`bptt_truncate`); standard practice for long sequences.
