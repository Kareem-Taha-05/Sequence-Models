A recurrent neural network written **from scratch with plain Python lists (no numpy)**, trained on a synthetic
"stock market" where tomorrow's price depends on the previous two days. The notebook starts by building the RNN
and then turns every claim of its theory sections into an experiment.

The only library used is `matplotlib` (for plots). Everything else (matrices, forward pass, BPTT, optimizers,
the MLP and the LSTM) is written by hand with the standard library.

---

## 1. What the notebook is about

**The environment.** Synthetic stocks prices that are discrete: Low (0.0), Medium (0.5), High (1.0). Four rules decide the next day:

| Rule | Last two days | Next day |
|---|---|---|
| The Floor | 0.0 -> 0.0 | usually 0.0 (80%) |
| The Surge | 0.0 -> 0.5 | 1.0 |
| The Crash | 1.0 -> 0.5 | 0.0 |
| The Ceiling | 1.0 -> 1.0 | usually 1.0 (80%) |

Every other pair is a random walk. This means the current price alone is not enough, the network has to
remember at least the previous day, and part of the data is random, so no model can reach zero error.

**The goal.** Check *experimentally* (not just by explaining) when and why a Vanilla RNN differs from a
feedforward network, measure the vanishing / exploding gradient problem, compare optimizers and hidden sizes,
implement an LSTM, and look inside the hidden state.

---

## 2. Notebook walkthrough, start to finish

### Part A: Building the RNN (sections 1 to 5)

| # | Section | What it contains |
|---|---|---|
| 1 | The data | The stock environment, its four transition rules, `generate_sequence`, and plots of three example companies. |
| 2 | Why RNNs? | Why feedforward networks struggle with sequences (fixed input size, no memory). |
| 3 | Anatomy of an RNN | Unrolling the loop and parameter sharing across time. |
| 4 | Forward pass maths | $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$ and the sigmoid output. |
| 5 | BPTT | Backpropagation through time, the product of Jacobians, vanishing and exploding gradients. |
| - | Code | Matrix helpers without numpy, the `VanillaRNN` class (forward + backward + gradient clipping), a training loop on one long sequence, and three hand-picked tests (surge, crash, ceiling). |

### Part B: Turning it into an experiment (sections 6 to 19)

**6. Setting up a proper experiment**
* One configuration cell holds every hyperparameter.
* Separate train / validation / test **sequences** (not a random split of windows). Neighbouring windows share
  5 of 6 days, so splitting windows would leak test data into training.
* Metrics: MSE, MAE and state accuracy (prediction snapped to Low / Medium / High).
* Baselines: **Mean**, **Persistence** (tomorrow = today), a **Markov-2 table** (best predictor that only sees
  yesterday and today) and an **Oracle** that knows the rules exactly. The oracle's error (test MSE 0.067, accuracy
  71%) is the noise floor. This explains why the first RNN's loss stopped at about 0.083 instead of going to zero.

**7. Making the code reusable**
* `TrainableRNN`: same forward pass and BPTT as `VanillaRNN`, but it *returns* the gradients instead of updating
  the weights, so any optimizer can be plugged in and the gradients can be inspected. A sanity check confirms it
  produces exactly the same update as the original (difference 0.0).
* `MLP` baseline with 11 hidden units, chosen so it has exactly the same number of parameters as the RNN (89).
* Four optimizers written from scratch: SGD, SGD with momentum, RMSprop, Adam.
* A reusable training loop (shuffled windows, optimizer object, train / validation loss for learning curves),
  experiment runner, plotting helpers and a results logger.

**8. Evaluating the RNN we already trained**
* The section 5 RNN is tested on 10 sequences it has never seen: test MSE 0.079 and 70.9% accuracy (oracle: 0.067
  and 71.0%).
* An **error-by-transition** table shows which rules were learned (surge and crash: nearly perfect) and where the
  remaining error is (the random transitions).
* Prediction vs. ground-truth plot.

**9. Baseline comparison: RNN vs. MLP**
* Same optimizer, same epochs, same number of parameters, 3 seeds.
* Controls for fairness: an MLP trained 3x longer, and an RNN given only the last-step training signal.
* Finding: the RNN wins at equal epochs, but mostly because it gets 6 training signals per window instead of 1.
  The converged comparison later is a tie, because the rules only need 2 days and the MLP sees 6.

**10. Experiment: optimizers**
* 4 optimizers x 3 learning rates (0.001, 0.01, 0.1), measuring convergence speed, stability (jitter), final loss
  and sensitivity to the learning rate.
* Finding: they end at the same noise floor; they differ mainly in how sensitive they are to the learning rate
  (SGD at 0.001 never converges, momentum at 0.1 diverges, Adam at 0.1 is unstable).

**11. Experiment: hidden layer size**
* 2, 4, 8, 16 and 32 hidden units, 2 seeds each: loss curves, parameters, overfitting gap and training time.
* Finding: 2 units underfit; from 4 units on everything is the same, so 8 was reasonable but 4 would do. Training
  time grows roughly with the square of the hidden size (2 s to 53 s).

**12. Experiment: generalization to longer sequences**
* Train on 6-day windows, test on sequences of 7, 20, 50 and 100 days: RNN run continuously, RNN restarted every
  6 days, MLP with a sliding window, and the oracle.
* Finding: the hidden state stays stable over 100 steps (no drift). This shows the RNN *supports* any length, not
  that it can *use* old information, because this environment only needs 2 days.

**13. Gradient analysis: vanishing and exploding gradients**
* Measures how much of the learning signal reaches a day *k* steps in the past, for sequence lengths 5 to 100.
* Compares initializations of the recurrent matrix (uniform scales 0.1 to 2.0, and orthogonal).
* Checks what gradient clipping does: it never touches vanishing gradients (0 entries clipped, signal unchanged).
* Trains with an exploding initialization with and without clipping. In this single short run, element-wise
  clipping did *not* help; the notebook flags this as a surprise and does not claim a cause.

**14. LSTM from scratch and comparison**
* Gate equations, the cell state, and why the forget bias starts at 1.
* A numerical **gradient check** of the LSTM and RNN backward passes (agreement to about 1e-8).
* Vanilla RNN vs LSTM (8 and 4 hidden units) vs MLP, 3 seeds, same optimizer, same data, same metrics.
* Gradient flow of the RNN hidden state vs the LSTM cell state, untrained and trained.
* Finding: on this environment the LSTM does not beat the RNN (it costs about 3.7x the parameters and time),
  because nothing here needs long memory. Its gradient flow is much better at initialization, but the decay is
  slowed, not removed.

**15. More sequential patterns** (RNN vs LSTM vs MLP on each)
* **Periodic:** an 8-day cycle that needs more than 2 days of memory. All three models reach 100% accuracy; even a
  2-unit RNN does.
* **Noisy:** Gaussian noise on the observed prices (sigma 0 to 0.5), clean-trained models and models trained with
  noise.
* **Anomalies:** rare spikes (2.0) and drops (-1.0), measuring the *excess error by days after the anomaly*. The
  RNN and LSTM recover in about 3 days; the MLP's recovery is bounded by its 6-day window.
* **Changing regimes:** the rules flip halfway through each sequence. A model trained on one regime never adapts;
  models trained with switches adapt partially.

**16. Hidden state analysis**
* Heatmap of all hidden units over a sequence, and each unit over time.
* The same current input (0.5) after different histories: the hidden states differ strongly (the network really
  uses its memory), while histories sharing the last two days stay close.
* Average hidden state for each (yesterday, today) pair.

**17. Results**
* One table with every experiment's parameters, train / validation / test MSE, MAE and accuracy.

**18. Limitations** and **19. Conclusion**
* What Vanilla RNNs struggle with, what these experiments could not show, a table mapping each theory claim to
  what was measured, and ideas for improvement.
