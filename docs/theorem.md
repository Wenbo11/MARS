# From Optimal Stopping to Margin Decomposition: A First-Principles Derivation

**Date:** 2026-03-08
**Status:** Design document

---

## 1. Setup

$N$ traces generate in parallel. At checkpoints $t = 1, \ldots, T$, each trace $j$ has an answer $a_j(t) \in \mathcal{A}(t) \cup \{\varnothing\}$ and weight $w_j > 0$, where $\varnothing$ denotes "no answer yet."

**Dynamic answer set.** Since the model is generative, the answer space is not known before inference. The observed answer set $\mathcal{A}(t)$ consists of all distinct answers produced by any trace up to checkpoint $t$ (including warmup traces). It grows monotonically: $\mathcal{A}(s) \subseteq \mathcal{A}(t)$ for $s \leq t$. We write $K(t) = |\mathcal{A}(t)|$. Two consequences:

1. **Theorem 1 (Section 3) is unaffected.** The decomposition $M_k(T) = R_k(t) + \Phi_k(t)$ is an algebraic identity for any specific challenger $k$ — it does not depend on enumerating $\mathcal{A}(T)$.

2. **The stopping condition ranges over $\mathcal{A}(T) \supseteq \mathcal{A}(t)$.** At time $t$, we must certify $M_k(T) > 0$ for all $k \neq L$ — including answers in $\mathcal{A}(T) \setminus \mathcal{A}(t)$ that have not yet appeared. The zero-vote challenger guard (Section 4.4) handles this: it checks a worst-case challenger with 0 current votes, which upper-bounds any novel answer's threat. For the union bound correction, we need $K \geq K(T)$; a safe $\mathcal{F}_0$-measurable upper bound is $K = N$ (each trace can produce at most one distinct answer).

The vote tally:

$$V_k(t) = \sum_{j:\, a_j(t) = k} w_j$$

The decision:

$$D(t) = \arg\max_k\, V_k(t)$$

We write $L = D(t)$ for the leader at checkpoint $t$ (suppressing the $t$ dependence when clear).

**Goal.** Stop at the earliest $t$ where we are confident $D(T) = D(t)$, subject to using at most $\delta$ error probability.

---

## 2. The Optimal Stopping Rule

### 2.1 The Sufficient Statistic

Define the **flip probability**:

$$\rho(t) = P\bigl(D(T) \neq D(t) \;\big|\; \mathcal{F}_t\bigr)$$

where $\mathcal{F}_t$ is all information available at checkpoint $t$.

**Proposition 1 (Optimality of threshold rule).** Among all stopping rules satisfying $P(D(\tau) \neq D(T)) \leq \delta$, the rule

$$\tau^* = \min\{t : \rho(t) \leq \delta\}$$

minimizes $E[C(\tau)]$ where $C(\tau) = \sum_j \text{tokens}_j(\tau)$.

*Proof.* Any feasible $\tau$ must satisfy $E[\rho(\tau)] \leq \delta$. Since $C$ is increasing in $t$ and $\rho(\tau) \leq \delta$ is the tightest per-realization constraint compatible with the expectation constraint, stopping at the first $t$ where it holds minimizes cost. (A Bellman argument with option value can do strictly better only in edge cases where the current $\rho(t) \leq \delta$ but waiting one step yields a dramatically tighter bound — see Appendix A.) $\square$

*Intuition.* $\rho(t)$ is the one number we need. Everything else — per-trace models, margins, CLT bounds — is machinery for computing or bounding $\rho(t)$.

### 2.2 Rewriting $\rho$ in Terms of Margins

$D(T) \neq D(t)$ iff some challenger overtakes the leader at $T$. For each challenger $k \neq L$, define the **margin**:

$$M_k(T) = V_L(T) - V_k(T)$$

Then:

$$\rho(t) = P\bigl(\exists\, k \neq L : M_k(T) \leq 0 \;\big|\; \mathcal{F}_t\bigr)$$

The flip probability reduces to the probability that any margin goes non-positive. This is the object we must compute or bound.

---

## 3. The Margin Decomposition Theorem

This section derives an exact decomposition of $M_k(T)$ into a part computable from per-trace switch probabilities and a part that requires destination knowledge. This decomposition is the structural reason for the gap between the oracle and estimated methods.

### 3.1 Partitioning Traces

For any challenger $k \neq L$, the votes at $T$ decompose by where each trace was at time $t$:

$$V_L(T) = \underbrace{\sum_{\substack{j:\, a_j(t) = L \\ a_j(T) = L}} w_j}_{\text{L-stayers}} \;+\; \underbrace{\sum_{\substack{j:\, a_j(t) \neq L \\ a_j(T) = L}} w_j}_{\text{switch-in to } L}$$

$$V_k(T) = \underbrace{\sum_{\substack{j:\, a_j(t) = k \\ a_j(T) = k}} w_j}_{\text{k-stayers}} \;+\; \underbrace{\sum_{\substack{j:\, a_j(t) \neq k \\ a_j(T) = k}} w_j}_{\text{switch-in to } k}$$

These are exact identities — no model, no approximation, just partitioning by $\{a_j(t), a_j(T)\}$.

### 3.2 The Decomposition

**Theorem 1 (Margin Decomposition).** For any checkpoint $t$ and challenger $k \neq L(t)$:

$$\boxed{M_k(T) = R_k(t) + \Phi_k(t)}$$

where:

$$R_k(t) = \sum_{\substack{j:\, a_j(t) = L \\ a_j(T) = a_j(t)}} w_j \;-\; \sum_{\substack{j:\, a_j(t) = k \\ a_j(T) = a_j(t)}} w_j$$

is the **retained margin** — the margin among traces that *do not change their answer*, and:

$$\Phi_k(t) = \sum_{\substack{j:\, a_j(T) = L \\ a_j(t) \neq L}} w_j \;-\; \sum_{\substack{j:\, a_j(T) = k \\ a_j(t) \neq k}} w_j$$

is the **destination effect** — the net switch-in advantage of the leader over challenger $k$.

*Proof.* Subtract $V_k(T)$ from $V_L(T)$:

$$M_k(T) = (\text{L-stayers} + \text{switch-in to } L) - (\text{k-stayers} + \text{switch-in to } k)$$
$$= (\text{L-stayers} - \text{k-stayers}) + (\text{switch-in to } L - \text{switch-in to } k) = R_k(t) + \Phi_k(t) \quad\square$$

*Intuition.* The final margin has two components: the margin among "loyalists" who don't change their answer ($R_k$), and the margin among "movers" who switch ($\Phi_k$). The first is about retention. The second is about attraction.

---

## 4. Methods in the Decomposition Framework

This section connects the margin decomposition (Theorem 1) to the stopping methods implemented in the codebase. We identify three distinct sources of uncertainty that any stopping method must handle, and show how each method addresses them.

### 4.1 Per-Trace Adversarial Cost

Every stopping method must ensure the leader's margin $M_k(T) > 0$ for all challengers $k$. Since the future is uncertain, we bound the worst-case margin damage from each trace switching its answer.

Define the **per-trace adversarial cost** for challenger $k$ — the worst-case change to $M_k$ when trace $j$ switches:

$$c_j^k = \begin{cases} 2w_j & \text{if } a_j(t) = L \quad \text{(leader trace: } L \text{ loses } w_j \text{, worst case } k \text{ gains } w_j\text{)} \\ -w_j & \text{if } a_j(t) = k \quad \text{(k-voter departure: } k \text{ always loses } w_j \text{, } M_k \text{ improves)} \\ w_j & \text{otherwise} \quad \text{(neutral trace: worst case } k \text{ gains } w_j\text{)} \end{cases}$$

**K-voter departure benefit.** When a trace voting for challenger $k$ switches to any answer $b \neq k$: $V_k$ loses $w_j$ (always), $V_L$ gains $w_j$ if $b = L$ or stays unchanged otherwise. So $\Delta M_k \geq +w_j$ regardless of destination. The worst case is $+w_j$ (switching to a non-leader), giving $c_j^k = -w_j$. This is tight — no destination information is needed, making it $\mathcal{F}_t$-measurable. See `deepdives/2026-03-14_structural_gap_analysis.md` Section 4.6 for the full proof.

The maximum damage trace $j$ can inflict on margin $M_k$ is $q_j \cdot c_j^k$, where $q_j = P(a_j(T) \neq a_j(t) \mid \mathcal{F}_t)$ is the switch probability. Note that for k-voters, this "damage" is negative — switching always helps the leader.

### 4.2 Three Sources of Uncertainty

Stopping at checkpoint $t$ requires ensuring $M_k(T) > 0$. Three distinct sources of uncertainty stand between the observable $M_k(t)$ and the unknown $M_k(T)$:

**(a) Destination uncertainty ($\Phi_k$).** The per-trace switch probability $q_j$ determines the **expected retained margin**:

$$\hat{R}_k(t) = \sum_{j:\, a_j(t) = L} (1 - q_j)\, w_j \;-\; \sum_{j:\, a_j(t) = k} (1 - q_j)\, w_j$$

But $M_k(T) = R_k(t) + \Phi_k(t)$, and $\Phi_k$ depends on **where** switching traces end up — information the q-model does not provide. $\Phi_k$ is invisible to any method that only uses per-trace switch probabilities. The NC stopping criterion (derived in Section 4.4) implicitly assumes $\Phi_k^{\text{NC}} = -(Q_{\text{total}} - Q_k)$ — all switch mass except k-voter departures is adversarial (Section 4.1). When the leader is the true winner, the actual $\Phi_k > 0$ typically (most switchers flow to the leader), so this remains deeply conservative.

**(b) Switching randomness.** Even with known $q_j$, the actual switching outcomes are random — each trace independently switches with probability $q_j$. The retained margin $R_k(t)$ is therefore a random variable. We need $P(R_k(t) + \Phi_k(t) > 0) \geq 1 - \delta$, which requires bounding the variance of $R_k$. This is addressed by either:
- **Hoeffding correction**: $w_{\max} \cdot \sqrt{2N \log(|A|/\delta)}$ — distribution-free, does not adapt to $q_j$
- **CLT correction**: $z \cdot \sqrt{\sum_j q_j(1 - q_j)(c_j^k)^2}$ — adapts to $q_j$, tighter when $q_j$ is small

**(c) q-model estimation error.** The true switch probabilities $q_j^*$ are unknown; we use estimates $\hat{q}_j$. The gap $\hat{q}_j - q_j^*$ propagates into the retained margin estimate. This is addressed by either:
- **Conservative upper bound**: $\hat{q}_j^{\text{upper}} = \hat{q}_j + k \cdot \text{SE}(\hat{q}_j)$ — inflates switch probs, making stopping harder
- **Relying on NC structural conservatism**: if the adversarial NC buffer (source (a)) is large enough, small q-model errors are absorbed

### 4.3 How Current Methods Address Each Source

The released methods are the NC (MARS) rows. Other rows (CLT correction, fixed-$\alpha$
margin) are shown for theoretical comparison; they are not part of the released code.

| Method | (a) Destination $\Phi_k$ | (b) Switching randomness | (c) q-model error |
|--------|--------------------------|--------------------------|---------------------|
| **Oracle** | Known exactly | N/A (deterministic) | N/A (known $q_j^* \in \{0,1\}$) |
| **NC, oracle $q$** (`*-mars-oracle`) | Adversarial worst-case | Hoeffding correction (off by default) | None ($q_j^*$ known) |
| **MARS, learned $q$** (`*-mars`) | Adversarial worst-case | NC structural buffer absorbs | NC structural buffer + $\gamma$ calibration absorb |
| _NC, CLT correction_ (theory) | Adversarial worst-case | CLT correction (tighter) | — |
| _Fixed-$\alpha$ margin_ (theory) | Adversarial worst-case | Implicit (uniform $\alpha$) | N/A ($\alpha$ is a hyperparameter) |

### 4.4 Method Descriptions

**Oracle.** Observes $\{a_j(T)\}_{j=1}^N$ — every trace's final answer. Computes $M_k(T)$ exactly:

$$\rho^{\text{oracle}}(t) = \mathbf{1}[D(T) \neq D(t)]$$

Stopping rule: $\tau^{\text{oracle}} = \min\{t : D(t) = D(T)\}$. Error rate: 0. Uses both $R_k$ and $\Phi_k$ exactly. For $\delta = 0$, this is the unique optimal stopping time (Proposition 1 with $\rho \in \{0,1\}$).

**NC (per-trace $q$).** Knows $q_j = P(a_j(T) \neq a_j(t) \mid \mathcal{F}_t)$ for each trace but not destinations. The stopping criterion for each challenger $k$ is:

$$M_k(t) > \sum_j q_j \cdot c_j^k + \text{correction}$$

This maps cleanly to the decomposition framework. Define $Q_a = \sum_{j:\,a_j(t)=a} q_j\, w_j$ (expected switch-out mass from answer $a$) and $Q_{\text{total}} = \sum_j q_j\, w_j$. The current margin decomposes as:

$$M_k(t) = \hat{R}_k(t) + (Q_L - Q_k)$$

and the adversarial damage (using the corrected $c_j^k$ from Section 4.1) decomposes as:

$$\sum_j q_j\, c_j^k = 2Q_L - Q_k + Q_o = (Q_L - Q_k) + (Q_{\text{total}} - Q_k)$$

where $Q_o = Q_{\text{total}} - Q_L - Q_k$. The switch-out differential $(Q_L - Q_k)$ appears on both sides and cancels:

$$\boxed{\hat{R}_k(t) > Q_{\text{total}} - Q_k + \text{correction}}$$

The threshold is per-challenger: it drops by $Q_k$ for each challenger $k$, reflecting that k-voter departures always help the leader's margin (Section 4.1).

In the decomposition framework, this is $\hat{R}_k(t) + \Phi_k^{\text{NC}}(t) > \text{correction}$, where:

$$\Phi_k^{\text{NC}}(t) = -(Q_{\text{total}} - Q_k)$$

NC's implicit destination effect assumption: all switch mass *except k-voter departures* is adversarial. The k-voter departure benefit is the tightest destination-free bound — it requires no knowledge of where switchers go. When the leader is the true winner, the actual $E[\Phi_k] > 0$ (positive — most switchers flow to the leader). The structural gap between NC and oracle is:

$$E[\Phi_k] - \Phi_k^{\text{NC}} = E[\Phi_k] + Q_{\text{total}} - Q_k$$

This gap is large at early positions where $Q_{\text{total}} \gg \hat{R}_k$ — many traces are expected to switch, and NC treats most of the switch mass as harmful. Closing this remaining gap requires destination knowledge $D_j(T)$, which is not $\mathcal{F}_t$-measurable (see `deepdives/2026-03-14_structural_gap_analysis.md` Section 5).

**CLT.** Same adversarial structure as NC (same $\hat{R}_k > Q_{\text{total}} - Q_k + \text{correction}$), but the correction adapts to $q_j$: $z \cdot \sqrt{\sum_j q_j(1-q_j)(c_j^k)^2}$. Note that the k-voter departure cost $c_j^k = -w_j$ contributes $q_j(1-q_j)w_j^2$ to the variance (since $(-w_j)^2 = w_j^2$), partially offsetting the threshold reduction. When $q_j$ is small, both $Q_{\text{total}}$ and the variance shrink, producing a tighter threshold than Hoeffding.

**AM (alpha-margin).** A special case of NC with uniform switch probability $q_j = \alpha$ for all active traces, $q_j = 0$ for finished traces.

**Proposition 2 (AM-NC equivalence).** The AM stopping criterion $M_k(t) \geq \alpha \cdot S_k(t)$ is equivalent to the NC criterion (without correction) with uniform q-model $\hat{q}_j = \alpha$ for active traces.

*Proof.* The NC criterion for challenger $k$ is $M_k > \sum_j q_j \cdot c_j^k$. With $q_j = \alpha$ for active traces and $q_j = 0$ for finished traces: $\sum_j q_j \cdot c_j^k = \alpha \sum_{j \in \mathcal{A}(t)} c_j^k = \alpha \cdot S_k(t)$, where $S_k(t) = \sum_{j \in \mathcal{A}(t)} c_j^k$ is the total adversarial cost of active traces. $\square$

**AM and the k-voter departure benefit.** In principle, AM can use the corrected costs from Section 4.1 ($c_j^k = -w_j$ for k-voters), which would give $S_k = w_{active,L} + w_{active} - 2w_{active,k}$. However, in practice AM uses the **loose** bound ($c_j^k = 0$ for k-voters, giving $S_k = w_{active,L} + w_{active} - w_{active,k}$). The reason: AM's hyperparameter $\alpha$ is calibrated against this cost structure. The k-voter correction shrinks $S_k$ significantly for strong challengers, making the threshold $\alpha \cdot S_k$ much smaller and requiring $\alpha$ recalibration. Since AM's appeal is simplicity (no model fitting), the loose bound is preferred — it is still valid and avoids introducing a calibration dependency.

AM requires no model fitting — just a user-specified $\alpha$. The tradeoff: no per-trace resolution (a high-confidence stable trace gets the same $q_j$ as a low-confidence oscillating one), and $\alpha$ calibration is critical. AM methods also include consensus stopping (OR logic): stop when the leader holds $\geq \tau$ fraction of total weight.

### 4.5 The Safety Budget and NC's Structural Buffer

The NC adversarial assumption creates a large safety buffer: it assumes all switching vote mass — except k-voter departures (Section 4.1) — goes to the worst-case challenger. In reality, when the leader is the correct answer, most of that mass goes TO the leader ($\Phi_k > 0$). This buffer implicitly absorbs both:

- **Switching randomness (b)**: small fluctuations in who actually switches are dwarfed by the adversarial overcount of damage
- **q-model error (c)**: moderate errors in $\hat{q}_j$ are absorbed because the adversarial treatment already vastly overestimates the damage

This is why `*-qm2-nc` methods (no conservative bound, no correction term) work in practice — NC's structural conservatism serves as a catch-all safety buffer. However, this comes at a cost: the same buffer that absorbs errors also prevents early stopping when it would be safe.

**K-voter departure and the safety budget.** The k-voter departure correction (Section 4.1) tightens NC's destination bound from $\Phi_k^{\text{NC}} = -Q_{\text{total}}$ to $\Phi_k^{\text{NC}} = -(Q_{\text{total}} - Q_k)$. Crucially, this preserves the safety buffer for sources (b) and (c): only k-voter costs change ($0 \to -w_j$), while leader-leaver ($2w_j$) and neutral-leaver ($w_j$) costs remain fully adversarial. The buffer shrinks by $Q_k$ per challenger — typically small because challengers have smaller vote shares than the leader.

**Implication for further reducing the structural gap.** Any approach that goes beyond the k-voter correction — replacing the remaining adversarial costs with tighter estimates — requires destination knowledge ($D_j(T)$) and removes the buffer that silently protects against (b) and (c). Such approaches must explicitly handle switching randomness and q-model error, which NC methods get "for free" from their structural conservatism. The remaining gap $E[\Phi_k] + Q_{\text{total}} - Q_k$ is information-theoretic: it requires knowing where switchers go, which is not $\mathcal{F}_t$-measurable.

### 4.6 Stopping Time Ordering

**Proposition 3 (Stopping time ordering).** On any realization where the estimated $\hat{R}_k$ is exact (perfect q-model) and $\Phi_k(t) \geq 0$ at the oracle stopping time:

$$\tau^{\text{oracle}} \leq \tau^{\text{NC}}$$

*Proof.* At $\tau^{\text{oracle}}$: $M_k(T) = R_k + \Phi_k > 0$ for all $k$. NC requires $\hat{R}_k > Q_{\text{total}} - Q_k \geq 0$. If $\Phi_k > 0$, we can have $M_k(T) > 0$ even when $\hat{R}_k \leq Q_{\text{total}} - Q_k$ (the destination effect compensates for the adversarial shortfall). In that case NC cannot certify safety and has not yet stopped. So $\tau^{\text{NC}} \geq \tau^{\text{oracle}}$. $\square$

**AM-NC ordering.** The position of $\tau^{\text{AM}}$ relative to $\tau^{\text{NC}}$ is not fixed. With small $\alpha$ (e.g., 0.01) and high true $\bar{q}$ (early checkpoints), AM can stop before NC — potentially incurring errors that NC would avoid. This is why AM requires $\alpha$ tuning while q-model methods adapt automatically.

---

## Appendix A. Relationship to the Bellman Optimal

### A.1 The Bellman Equation

The fully optimal stopping rule solves:

$$V(s_t) = \min\Bigl\{\underbrace{C(t) + \lambda \cdot \rho(t)}_{\text{stop}},\;\; \underbrace{E[V(s_{t+1}) \mid s_t]}_{\text{continue}}\Bigr\}$$

where $\lambda$ encodes the error-vs-cost tradeoff (dual variable for the constraint $P(\text{error}) \leq \delta$).

### A.2 Oracle + Bellman

With oracle information, $\rho(t) \in \{0, 1\}$. When $\rho(t) = 0$ (current decision is correct):

$$V(s_t) = \min\{C(t),\; E[V(s_{t+1}) \mid s_t]\}$$

Since $C(t) < C(t+1) \leq E[V(s_{t+1})]$, stopping is optimal. Therefore:

**Proposition 4.** With oracle information and hard constraint ($\delta = 0$):

$$\tau^{\text{Bellman-oracle}} = \tau^{\text{oracle}}$$

The threshold rule and the Bellman optimal coincide — there is no option value when the current decision is known to be correct.

### A.3 Soft Constraint ($\delta > 0$)

With $\delta > 0$, the Bellman optimal may stop at $t$ where $\rho(t) = 1$ (accepting an error) if the token savings are large enough. Specifically, on realizations where $\tau^{\text{oracle}}$ is near $T$ (the oracle would barely save anything), the Bellman optimal might stop much earlier, spending part of its error budget for large savings.

$$E[C(\tau^{\text{Bellman-oracle}})] \leq E[C(\tau^{\text{oracle}})]$$

with equality at $\delta = 0$.

### A.4 Practical Irrelevance

The Bellman gain over oracle is bounded by $\delta \cdot C_{\text{max}}$ (at most $\delta$ fraction of questions contribute savings, each saving at most $C_{\text{max}}$ tokens). For $\delta = 0.05$, this is at most a 5% improvement — much smaller than the gap between oracle and estimated methods.

The real bottleneck is not the Bellman vs. oracle gap. It is the oracle vs. estimated gap, which is driven by the destination effect $\Phi_k$.

---

## Appendix B. Summary of Gaps and Method Hierarchy

### B.1 Summary of Gaps

$$E[C(\tau^{\text{Bellman}})] \;\leq\; E[C(\tau^{\text{oracle}})] \;\leq\; E[C(\tau^{\text{NC}})]$$

| Gap | Between | Driven by | Magnitude |
|-----|---------|-----------|-----------|
| Option value | Bellman $\leftrightarrow$ Oracle | Error budget allocation | $\leq \delta \cdot C_{\text{max}}$ (small) |
| Structural (destination) | Oracle $\leftrightarrow$ NC | $\Phi_k$ bounded adversarially | Information-theoretic (requires $D_j(T)$) |
| q-model quality | NC (oracle q) $\leftrightarrow$ NC (estimated q) | $\hat{q}_j$ vs $q_j$ | Moderate (dataset-dependent) |
| Per-trace resolution | NC $\leftrightarrow$ AM | Uniform $\alpha$ vs per-trace $q_j$ | Depends on trace heterogeneity |

The structural gap is the dominant component. The k-voter departure correction (Section 4.1) captures the only destination-free tightening; the remaining gap requires knowing where switchers go ($\mathcal{F}_t$-immeasurable). See `deepdives/2026-03-14_structural_gap_analysis.md` for the full analysis.

The position of $\tau^{\text{AM}}$ relative to $\tau^{\text{NC}}$ is not fixed — it depends on $\alpha$ vs the actual $\hat{q}_j$ distribution (see Section 4.6).

### B.2 The Key Equation

Everything reduces to one decomposition:

$$\boxed{M_k(T) = \underbrace{R_k(t)}_{\text{per-trace } q} + \underbrace{\Phi_k(t)}_{\text{destination effect}}}$$

### B.3 Method Hierarchy

Each method makes progressively stronger assumptions to bound $M_k(T)$:

| Method | What it computes | What it assumes about $\Phi_k$ | q-model |
|--------|-----------------|-------------------------------|---------|
| **Oracle** | $R_k + \Phi_k$ exactly | Known | N/A (observes $a_j(T)$) |
| **NC** | $\hat{R}_k$ and per-answer $Q_k$ | $\Phi_k \geq -(Q_{\text{total}} - Q_k)$ | Per-trace $q_j$ |
| **AM** | $M_k - \alpha \cdot S_k$ | $\Phi_k \geq -Q_{\text{total}}$ (loose bound) | Uniform $q_j = \alpha$ |

NC uses the tight adversarial bound from Section 4.1 (k-voter departures help, everything else worst-case). AM uses the loose bound ($c_j^k = 0$ for k-voters) to preserve $\alpha$ calibration. Moving down the table: less information $\Rightarrow$ later stopping $\Rightarrow$ fewer token savings (in the well-calibrated regime).
