# Decoded Exam Material — Inteligență Artificială

This document is a single unified source for the exam material in this directory.
It transcribes both the **photographs** (`PHOTO-2026-05-26-20-35-*.jpg`) and the
**`Simulare.pdf`** mock exam. The photos are low-quality shots — partly screenshots
of a Moodle-style online exam, partly projector photos with the instructor's
handwritten worked solutions in red, and partly photos of additional exam tickets
(some typed, some handwritten). Each region was cropped and enlarged to recover the
handwriting; the result is transcribed below so the images themselves are no longer
needed.

**Course:** Inteligență Artificială (Denis Enăchescu, Licență I.D., an II,
University of Bucharest).

**Exam format reminder:** each ticket has **5 questions** (one ticket, "Bilet 10",
has 6) covering the whole syllabus, each marked in the margin with its point value
(1–3 points), summing to 10.

**What the photos contain.** The first four photos all capture **one** exam page
(the "on-screen exam") in increasing detail, including the instructor's red model
solutions. The remaining photos are **six further exam tickets** (B12, B18, Bilet
10, Examen IA, B11, B15) — questions only, no solutions. The same topics recur
across tickets. The structure of this document is:

1. The on-screen exam, question by question, **with worked solutions** (the only
   place solutions appear).
2. Each additional ticket, listed separately, then the `Simulare.pdf` mock exam
   (which also lists the exact course pages for each answer).
3. A consolidated **master list of distinct question types** (below) — the single
   most useful section for revision, since it deduplicates everything.

---

## Master list of distinct question types (deduplicated)

Across all sources — the seven photographed tickets plus the `Simulare.pdf` mock
exam — these are the distinct questions that appear (with the tickets each one
shows up in). Items marked ★ have a worked solution transcribed later in this
document (from the on-screen exam's red annotations or the Simulare grading scheme).

| # | Topic | Appears in |
|---|-------|------------|
| A ★ | Define ERM **consistency** + illustrate graphically | on-screen Q1, B18·1, Bilet10·2, B15·1, Simulare·4 |
| B ★ | What assumptions must training examples satisfy? (**i.i.d.**) | on-screen Q2, B18·2, B11·2, B15·2 |
| C ★ | **Chain network N** `x→a→f→b→f→y`: write hypothesis space H + particularize **off-line** backprop with SSE | on-screen Q3, B12·3, B15·3 |
| D ★ | Which perceptron **components change** during training + state a learning rule (**Rosenblatt rule**; weights w, bias b) | on-screen Q4, Examen IA·4 |
| E ★ | **Purelin network N** `x→w1→purelin→w2→purelin→y` (bias b2): can it implement f(x)=x exactly? (yes iff w₂w₁=1, b₂=0) | on-screen Q5, B12·5, B18·5 |
| E′ | Same purelin network: particularize **batch** backprop with SSE | B11·5 |
| F | **Rosenblatt perceptron** for scalar inputs: definition + diagram; which sets can it learn (linearly separable) | B12·1, Examen IA·1, B11·1 |
| G ★ | **Training vs. validation set** + error-curve graph + **early stopping** moment | B12·2, Simulare·1 |
| H | **K₀ / max-epochs bound** for a Rosenblatt perceptron to learn a linearly separable set; which point categories hinder learning | B12·4, B18·4, Bilet10·3, B11·4, B15·4 |
| I | Construct a perceptron for the logical **AND** function + verify | B18·3, Bilet10·5, Examen IA·3 |
| J | **Total supervised-learning error** formula for noise-free labels | Bilet10·1 |
| K | **Two methods to speed up backprop** + justify | Bilet10·4, Examen IA·5 |
| L | Show **saturation is eliminated** in a linear-transfer MLP (RPM) trained with SSE + backprop | Bilet10·6 |
| M | **Two methods to estimate the real risk** (+ formulas); how real vs. empirical risk behave as hypothesis-space complexity grows | Examen IA·2 |
| N | **Network R** `x→w1→logsig→w2→purelin→y` (biases b1, b2): compute **∂J/∂w1** for the interactive (online) learning strategy, SSE | B11·3 |
| O | **True/False**: backprop requires transfer functions to be signum/Heaviside? (False — backprop needs differentiable activations) | B15·5 |
| P ★ | Construct a perceptron for the Boolean unary **NOT** function + justify weights + verify | Simulare·2 |
| Q ★ | Define the **saturation / flat-spot** effect + give a transfer function free of it (**purelin**) + justify | Simulare·3 |
| R ★ | **Parallel two-perceptron network** (one input x, two outputs y₁,y₂): write hypothesis space H + particularize **online** gradient descent (state the performance function) | Simulare·5 |

---

## The on-screen exam — 5 questions (with worked solutions)

The five questions (and their point values, from the "Marcat din X,00" labels) are:

| # | Points | Topic |
|---|--------|-------|
| 1 | 2,00 | Consistency of the empirical-risk-minimization principle (definition + graph) |
| 2 | 1,00 | Hypotheses the training examples must satisfy |
| 3 | 3,00 | A 2-perceptron chain network N: hypothesis space H + off-line backprop |
| 4 | 2,00 | Which perceptron components change during training + a learning rule |
| 5 | 2,00 | A 2-layer linear (purelin) network: can it implement f(x)=x exactly? |

Total = 10 points.

---

## Question 1 — Consistency of empirical risk minimization (2 p)

**Statement:**
> Definiți consistența principiului minimizării riscului empiric. Ilustrați
> grafic această definiție.
> *(Define the consistency of the empirical-risk-minimization principle.
> Illustrate this definition graphically.)*

**Model solution.**

Definitions of the relevant minimizers:

- ĥₙ = argmin_{h∈H} Rₙ(h) — the empirical-risk minimizer.
- ĥ_{Sₘ} = argmin_{h∈H} R_emp(h) — the hypothesis that minimizes the empirical
  risk over a training sample Sₘ.
- card(Sₘ) = m — the sample has cardinality m.

**Graphical illustration** (curves plotted against the sample size):

- A **decreasing** curve labeled **Rₙ(ĥ_{S,n})** — the empirical risk of the
  empirical minimizer.
- An **increasing** curve labeled **R(ĥ_{S,m})** — the true (actual) risk of the
  empirical minimizer.
- Both curves converge toward the optimal risk **R(ĥ\*)** as m → ∞.

ERM is **consistent** when both the empirical risk and the true risk of the
empirical minimizer converge to the minimal achievable risk as the sample size
grows.

*Study reference noted in the margin: lecture **L02**, p. 23.*

---

## Question 2 — Assumptions on the training examples (1 p)

**Statement:**
> Ce ipoteze trebuie să satisfacă exemplele mulțimii de antrenare?
> *(What assumptions must the examples of the training set satisfy?)*

**Model solution:** the examples must be **i.i.d.** — independent and identically
distributed.

*Study reference noted in the margin: lecture **L02**, p. 10.*

---

## Question 3 — Two-perceptron chain network N (3 p)

**Network N** (single real input, two perceptrons in series, transfer function
f on each neuron, no biases):

```
        a              b
  x ●───────▶( f )───────▶( f )───────▶ y
```

- weight `a` from input x into the first neuron,
- weight `b` between the first and second neurons.

**Trained on the set** S = { (xᵢ, tᵢ) }, with xᵢ and tᵢ real numbers, i = 1,…,n.

**Statement:**
> 1. Scrieți spațiul H al ipotezelor implementat de rețeaua (N).
> 2. Particularizați algoritmul backprop, forma "off line", cu funcția eroare
>    "suma pătratelor erorilor", pentru rețeaua (N).
> *(1. Write the hypothesis space H implemented by network N. 2. Particularize
> the back-propagation algorithm, off-line form, with the sum-of-squared-errors
> error function, for network N.)*

**Model solution.**

*Part 1 — hypothesis space* (with the transfer function chosen as `tansig`):

> H = { f(a,b,·) : ℝ → ℝ | f(a,b;x) = tansig( b · tansig(a·x) ), a, b ∈ ℝ }

*Part 2 — off-line backprop with sum-of-squared-errors:*

Forward pass for example i:

> net₁,ᵢ = a · xᵢ          z₁,ᵢ = f(net₁,ᵢ)
> net₂,ᵢ = b · z₁,ᵢ        yᵢ  = f(net₂,ᵢ)

Error (performance) function — sum of squared errors:

> J(a,b) = ½ · Σᵢ₌₁ⁿ (tᵢ − yᵢ)²

Transfer function and its derivative (hyperbolic-tangent sigmoid):

> f(x) = tansig(x)
> f′(x) = 1 − f²(x)

(The off-line / batch form means the weights a and b are updated once per epoch
using the gradient of J summed over all n examples.)

*Study reference noted in the margin: lecture **L09**.*

---

## Question 4 — Components modified during training (2 p)

**Statement:**
> Ce componente ale perceptronului se modifică în timpul antrenării? Formulați
> și explicați o regulă de învățare care modifică aceste componente.
> *(Which components of the perceptron are modified during training? Formulate
> and explain a learning rule that modifies these components.)*

**Model solution:** the components that change are the **weights w** and the
**bias b**. The learning rule given is **Rosenblatt's rule** (the perceptron
learning rule), to be exemplified.

*Study references noted in the margin: lectures **L05 / L07**.*

---

## Question 5 — Two-layer linear network; can it implement f(x)=x? (2 p)

**Network N** (two `purelin` — linear — neurons in series):

```
        w1                    w2
  x ●────────▶[ purelin ]①────────▶[ purelin ]②────────▶ y
                                          ▲
                                          │ b2
                                          1
```

- neuron ①: linear (`purelin`), weight `w1`, no bias;
- neuron ②: linear (`purelin`), weight `w2`, bias `b2` (from the constant input 1).

**Statement:**
> Poate rețeaua N să implementeze, fără eroare, funcția f(x)=x? Justificați.
> *(Can network N implement the function f(x)=x without error? Justify.)*

**Model solution.**

> net₁ = w₁·x ;  purelin(net₁) = w₁·x ≡ z₁
> net₂ = w₂·z₁ + b₂ ;  purelin(net₂) = w₂·z₁ + b₂
>       = w₂·(w₁·x) + b₂

Therefore the network output is y = w₂·w₁·x + b₂, so **y ≡ x** holds if and only
if:

> **w₂ · w₁ = 1   and   b₂ = 0.**

**Answer: YES** — network N can implement f(x)=x exactly, but only under the two
conditions above.

*Study reference noted in the margin: lecture **L10**.*

---

## Ticket "B 12" — a second, handwritten exam ticket

`PHOTO-2026-05-26-20-35-56-1.jpg` is a separate handwritten page (blue ink, graph
paper) headed **"B 12"** — the student's hand-copied exam ticket. It contains
**questions only (no solutions)**. Questions 3 and 5 are the same network
problems as the on-screen exam (Q3 chain network, Q5 purelin network, transcribed
above); questions 1, 2 and 4 cover additional topics:

**B12 · 1.**
a) Definiți perceptronul Rosenblatt pentru cazul intrărilor scalare. Desenați
diagrama acestuia. *(Define the Rosenblatt perceptron for scalar inputs; draw its
diagram.)*
b) Ce mulțimi poate învăța perceptronul de la punctul a)? *(What sets can that
perceptron learn?)*

**B12 · 2.** Definiți, în cazul învățării supervizate, mulțimea de antrenare și
mulțimea de validare. Desenați și explicați graficul erorii de antrenare și
validare. Precizați momentul early-stopping. *(In supervised learning, define the
training set and the validation set. Draw and explain the training/validation
error graph. Specify the early-stopping moment.)*

**B12 · 3.** Identical to the on-screen Q3 (chain network N: `x →a→ f →b→ f → y`,
trained on S = {(xᵢ,tᵢ)}): write the hypothesis space H; particularize off-line
backprop with the sum-of-squared-errors cost.

**B12 · 4.**
a) Definiți perceptronul Rosenblatt pentru cazul intrărilor scalare. Desenați
diagrama acestuia. *(Identical wording to B12·1a.)*
b) Dacă o mulțime poate fi învățată de perceptronul de la punctul a), care este
nr. maxim de epoci necesar? *(If a set can be learned by that perceptron, what is
the maximum number of epochs needed? — i.e. the perceptron convergence theorem.)*

**B12 · 5.** Identical to the on-screen Q5 (two `purelin` neurons in series, bias
b2 on the second): can N implement f(x)=x without error? Justify.

> This ticket adds the following topics not present in the on-screen exam:
> the Rosenblatt perceptron for scalar inputs (definition + diagram), which sets
> it can learn (linearly separable sets), training-vs-validation sets with the
> error-curve / early-stopping graph, and the maximum number of epochs
> (perceptron convergence).

---

## Ticket "B18" — a third exam ticket

`PHOTO-2026-05-26-20-35-56-2.jpg` is a clean typed printout headed **"B18"**
(questions only, no solutions), with point values in the margin:

**B18 · 1.** (2,00) Definiți consistența principiului minimizării riscului
empiric. Ilustrați grafic această definiție. *(Same as the on-screen Q1 — ERM
consistency, definition + graph.)*

**B18 · 2.** (1,00) Ce ipoteze trebuie să satisfacă exemplele mulțimii de
antrenare? *(Same as the on-screen Q2 — the i.i.d. assumption.)*

**B18 · 3.** (3,00) **Construiți perceptronul care implementează funcția logică
AND. Verificați corectitudinea funcționării acestuia.** *(Construct the perceptron
that implements the logical AND function. Verify the correctness of its
operation.)* — a Boolean-function perceptron exercise (analogous to the
NOT-function exercise in `Simulare.pdf`).

**B18 · 4.** (2,00)
a) Definiți perceptronul Rosenblatt pentru cazul intrărilor scalare. Desenați
diagrama acestuia.
b) Dacă o mulțime poate fi învățată de perceptronul de la pctul a), care este
numărul maxim de epoci necesar? *(Same as B12·4 — Rosenblatt perceptron + maximum
number of epochs / perceptron convergence.)*

**B18 · 5.** (2,00, flagged) Fie rețeaua (N): two `purelin` neurons in series
(`x →w1→ purelin →w2→ purelin → y`, bias b2 on the second neuron). Poate rețeaua N
să implementeze, fără eroare, funcția f(x)=x? Justificați. *(Same purelin network
as the on-screen Q5 / B12·5.)*

> New topic added by B18: constructing a perceptron for the logical **AND**
> function and verifying it.

---

## Ticket "Bilet 10" — a fourth exam ticket (6 questions)

`PHOTO-2026-05-26-20-35-57.jpg` is a handwritten page headed **"Bilet 10 – IA"**
(questions only). Unlike the others it has **six** questions:

**Bilet 10 · 1.** Explicați formula erorii totale de învățare supervizată în cazul
etichetelor neperturbate de zgomot. *(Explain the formula of the total
supervised-learning error in the case of labels unperturbed by noise.)*

**Bilet 10 · 2.** Definiți consistența principiului minimizării riscului empiric.
Ilustrați grafic această definiție. *(ERM consistency — same as the on-screen Q1 /
B18·1.)*

**Bilet 10 · 3.** Scrieți marginea **K₀** a numărului maxim de iterații necesar
unui perceptron Rosenblatt pentru a învăța o mulțime liniar separabilă. Ce
categorii de puncte îngreunează învățarea? Justificați. *(Write the bound K₀ on the
maximum number of iterations a Rosenblatt perceptron needs to learn a linearly
separable set. Which categories of points make learning harder? Justify.)*

**Bilet 10 · 4.** Prezentați 2 metode de creștere a vitezei algoritmului backprop.
Justificați pe scurt cum se îmbunătățește prin aceste metode viteza algoritmului.
*(Present 2 methods for speeding up the backprop algorithm. Justify briefly how
each improves the speed.)*

**Bilet 10 · 5.** Construiți perceptronul care implementează funcția logică AND.
Verificați corectitudinea funcționării acestuia. *(AND perceptron — same as
B18·3.)*

**Bilet 10 · 6.** Arătați [că] pentru o rețea **RPM** (rețea de perceptroni
multistrat / MLP) cu funcția de transfer liniară și funcție de eroare "suma
pătratelor erorilor", antrenată backprop, problema saturării perceptronilor de pe
stratul de ieșire și de pe straturile ascunse este eliminată. *(Show that for an
MLP with a linear transfer function and the sum-of-squared-errors cost, trained
with backprop, the saturation problem of the output-layer and hidden-layer
perceptrons is eliminated.)* — a small trailing symbol after "liniară" (≈ σ/Δ)
could not be fully resolved; the meaning is clearly "linear transfer function."

> New topics added by Bilet 10: the total-error formula for noise-free labels;
> the K₀ iteration bound for Rosenblatt-perceptron learning of a linearly
> separable set (and which points hinder learning); two methods for speeding up
> backprop; and the elimination of the saturation (flat-spot) problem in a
> linear-transfer MLP trained with SSE.

---

## Ticket "Examen IA" — a fifth exam ticket

`PHOTO-2026-05-26-20-35-57-1.jpg` is a (neatly) handwritten page headed **"Examen
IA"** (questions only), 5 questions:

**Examen IA · 1.** a) Definiți perceptronul Rosenblatt pt. cazul intrărilor
scalare. Desenați diagrama acestuia. b) Ce mulțimi poate învăța perceptronul de la
a)? *(Rosenblatt perceptron for scalar inputs + diagram; what sets it can learn —
same as B12·1.)*

**Examen IA · 2.**
a) Specificați 2 metode de estimare a riscului real. Dați formulele și explicați
pe scurt metodele. *(Specify 2 methods for estimating the real risk; give the
formulas and briefly explain the methods.)*
b) Cum variază riscul real și riscul empiric când complexitatea algoritmică a
spațiului de ipoteze crește? Specificați relația între riscul real și riscul
empiric în acest caz. *(How do the real risk and the empirical risk vary as the
algorithmic complexity of the hypothesis space increases? Specify the relation
between real and empirical risk in that case.)*

**Examen IA · 3.** Construiți perceptronul care implementează funcția logică AND.
Verificați corectitudinea funcționării acestuia. *(AND perceptron — same as
B18·3.)*

**Examen IA · 4.** Ce componente ale perceptronului se modifică în timpul
antrenării? Formulați și explicați o regulă de învățare care modifică aceste
componente. *(Components modified during training + a learning rule — same as the
on-screen Q4.)*

**Examen IA · 5.** Prezentați 2 metode de creștere a vitezei algoritmului
backprop. Justificați pe scurt cum se îmbunătățește, prin aceste metode, viteza
algoritmului. *(Two backprop speed-up methods — same as Bilet 10·4.)*

> New topics added by Examen IA (both in Q2): two methods for estimating the real
> risk (with formulas), and the behaviour of the real vs. empirical risk as the
> complexity of the hypothesis space grows.

---

## Ticket "B11" — a sixth exam ticket (typed)

`PHOTO-2026-05-26-20-35-57-2.jpg` is a clean typed printout headed **"B11"**
(questions only), with point values in the margin:

**B11 · 1.** (2,00) a) Definiți perceptronul Rosenblatt pentru cazul intrărilor
scalare. Desenați diagrama acestuia. b) Ce mulțimi poate învăța perceptronul de la
pctul a)? *(Rosenblatt perceptron for scalar inputs + diagram; what sets it can
learn — same as B12·1 / Examen IA·1.)*

**B11 · 2.** (1,00) Ce ipoteze trebuie să satisfacă exemplele mulțimii de
antrenare? *(i.i.d. — same as the on-screen Q2.)*

**B11 · 3.** (3,00, flagged) **Fie rețeaua R** (a new network, distinct from the
chain network N): `x →w1→ [logsig] →w2→ [purelin] → y`, with bias **b1** on the
logsig neuron and bias **b2** on the purelin neuron (each from a constant input 1).
> Calculați derivata parțială a lui J în funcție de w1, unde s-a notat cu J funcția
> de performanță "suma pătratelor erorilor", pentru strategia de învățare
> interactivă.
*(Compute the partial derivative ∂J/∂w1, where J is the sum-of-squared-errors
performance function, for the interactive — i.e. online — learning strategy.)*

**B11 · 4.** (2,00) Scrieți marginea k₀ a numărului maxim de iterații necesar unui
perceptron Rosenblatt pentru a învăța o mulțime liniar separabilă. Ce categorii de
puncte îngreunează învățarea? Justificați. *(Same as Bilet 10·3.)*

**B11 · 5.** (2,00, flagged) **Fie rețeaua (N):** `x →w1→ [purelin] →w2→ [purelin]
→ y`, bias b2 on the second neuron. Particularizați algoritmul backprop, forma
**batch**, cu funcția de performanță "suma pătratelor erorilor", pentru rețeaua N.
*(Particularize the backprop algorithm, batch (= off-line) form, with the SSE cost,
for the purelin network N.)*

> New topics added by B11: the network **R** = logsig → purelin (with biases) and
> the explicit computation of ∂J/∂w1 under the interactive/online learning
> strategy; and particularizing **batch** backprop with SSE for the two-purelin
> network N.

---

## Ticket "B15" — a seventh exam ticket (typed)

`PHOTO-2026-05-26-20-35-57-3.jpg` is a typed printout; the ticket label is washed
out in the photo but appears to read **"B15"** (number not fully certain). Point
values in the margin:

**B15 · 1.** (2,00) Definiți consistența principiului minimizării riscului empiric.
Ilustrați grafic această definiție. *(ERM consistency — same as the on-screen Q1.)*

**B15 · 2.** (1,00) Ce ipoteze trebuie să satisfacă exemplele mulțimii de
antrenare? *(i.i.d. — same as the on-screen Q2.)*

**B15 · 3.** (3,00, flagged) Fie rețeaua (N) `x →a→ f →b→ f → y`, antrenată pe
mulțimea S = {(xᵢ,tᵢ)} cu xᵢ și tᵢ numere reale, i = 1,…,n. 1. Scrieți spațiul H al
ipotezelor implementat de rețeaua (N). 2. Particularizați algoritmul backprop,
forma "off line", cu funcția eroare "suma pătratelor erorilor", pentru rețeaua (N).
*(The chain network — same as the on-screen Q3.)*

**B15 · 4.** (2,00) Scrieți marginea k₀ a numărului maxim de iterații necesar unui
perceptron Rosenblatt pentru a învăța o mulțime liniar separabilă. Ce categorii de
puncte îngreunează învățarea? Justificați. *(Same as Bilet 10·3.)*

**B15 · 5.** **Adevărat / Fals:** Metoda de învățare backprop cere ca funcțiile de
transfer ale perceptronilor să fie signum sau Heaviside. Justificare. *(True/False:
the backprop learning method requires the perceptrons' transfer functions to be
signum or Heaviside. Justify. — Expected answer: FALSE, because backprop relies on
the gradient and therefore needs differentiable transfer functions; signum /
Heaviside are not differentiable.)*

> New topic added by B15: a True/False item on whether backprop requires
> signum/Heaviside (step) transfer functions.

---

## "Simulare" — the official mock exam (`Simulare.pdf`)

`Simulare.pdf` is the instructor's **mock exam**: 5 questions, each followed by a
**grading breakdown** and **references to the course materials** where the answer
is found. The grading references are reproduced because they tell you exactly which
course/synthesis pages to study.

**Simulare · 1.** (2 p) Definiți, în cazul învățării supervizate, mulțimea de
antrenare și mulțimea de validare. Desenați și explicați graficul erorii de
antrenare și de validare. Precizați momentul early stopping. *(Define the training
set and the validation set in supervised learning. Draw and explain the
training/validation error graph. Specify the early-stopping moment.)*
Grading: definitions of the sets — 1 p (course *Estimation of the Hypothesis Real
Risk*); graph + explanation — 0.5 p (*The Bias-Variance Tradeoff*, p. 14 or p. 22,
"over-fitting"); identifying the early-stopping moment and why it is that point —
0.5 p (same pages). *(Same topic as B12·2.)*

**Simulare · 2.** (3 p) Funcția booleană unară **non X** ia valoarea 0 dacă x=1 și
valoarea 1 dacă x=0. Construiți perceptronul care implementează funcția non x
(schema, ponderi, funcție de integrare, funcție de transfer). Justificați modul în
care ați calculat ponderile. Verificați funcționarea corectă a perceptronului.
*(The unary boolean NOT function takes value 0 if x=1 and 1 if x=0. Construct the
perceptron implementing NOT x — diagram, weights, integration function, transfer
function. Justify how you computed the weights. Verify it works correctly.)*
Grading / solution sketch:
- Construction (diagram, weights, integration & transfer functions) — *Sinteza
  MLP*, drawing p. 33.
- Justification of the weights — *Sinteza perceptronului* p. 11. For the perceptron
  to implement NOT correctly we need a pair (w, b) with:
  - x = 0  →  w·0 + b ≥ 0  ⇒ output 1
  - x = 1  →  w·1 + b < 0  ⇒ output 0

  (The PDF writes the two integration inequalities `w*x+b >=0` and `w*x+b < 0`;
  the solution is **not unique** — any (w, b) satisfying b ≥ 0 and w + b < 0
  works, e.g. w = −1, b = 0.) *(Sinteza MLP p. 34 — verification.)*
- Verify correct operation for each value of x — *Sinteza perceptronului* p. 12.

**Simulare · 3.** (1 p) Definiți efectul de saturație. Indicați o funcție de
transfer a perceptronului care, pentru algoritmul de învățare al gradientului
descendent, nu prezintă acest efect. Justificați. *(Define the saturation effect.
Indicate a perceptron transfer function that does not exhibit this effect for the
gradient-descent learning algorithm. Justify.)*
Grading / solution sketch:
- Correct definition — 0.5 p (*Sinteza perceptronului* p. 40 — "flat spot";
  begins "Se numește efect de saturație …"). The saturation (flat-spot) effect:
  when a neuron's net input is large in magnitude, a squashing transfer function
  (logsig/tansig) has a derivative ≈ 0, so the gradient vanishes and learning
  stalls.
- Correctly defined transfer function — *Sinteza perceptronului* p. 12 — **purelin**
  (the linear function, whose derivative is a nonzero constant, so it has no flat
  spot).

**Simulare · 4.** (2 p) Definiți consistența principiului minimizării riscului
empiric. Ilustrați grafic această definiție. *(ERM consistency, definition +
graph.)* Grading: *Modelul supervizat* p. 14 (both the definition and the graph).
*(Same topic as the on-screen Q1 — see that section for the worked answer.)*

**Simulare · 5.** (2 p) **Fie rețeaua de perceptroni** below — a network with one
input x and **two perceptrons in parallel** (two outputs y₁, y₂):

```
            w11          ┌───┐
   x ●───────────────────▶│ f │──────▶ y1
       \         (bias w10 ┘ ▲ │
        \                 1 ─┘ └───┐
         \   w21          ┌───┐
          \───────────────▶│ f │──────▶ y2
                  (bias w20 ┘ ▲
                          1 ─┘
```
i.e. neuron 1: net₁ = w11·x + w10, y₁ = f(net₁); neuron 2: net₂ = w21·x + w20,
y₂ = f(net₂); with x, w10, w11, w20, w21 ∈ ℝ.
a) Scrieți spațiul de ipoteze H implementat de rețea. *(Write the hypothesis space
H.)* — *Modelul supervizat* p. 1; *Sinteza perceptronului* p. 12 — **logsig**
(suggesting f = logsig).
b) Particularizați algoritmul de gradient descendent pentru strategia de învățare
**interactivă** (online) la rețeaua de mai sus (specificați funcția de performanță
folosită). *(Particularize gradient descent for the interactive/online learning
strategy; specify the performance function.)* — *Sinteza perceptronului* p. 11,
p. 38 for the gradient algorithm, formula 2.53.

> New question types introduced by the Simulare that do not appear in the
> photographed tickets: the Boolean **NOT** perceptron (P), the **saturation /
> flat-spot** effect + a transfer function free of it (Q), and a **parallel
> two-perceptron network** with two outputs (R). It also lists the exact
> course/synthesis pages for every answer.

---

## Margin study references (student's handwriting)

Next to each question the student noted where the answer is in their synthesis
notes:

| Question | Reference |
|----------|-----------|
| Q5 | L10 |
| Q4 | L05 / L07 |
| Q3 | L09 |
| Q2 | L02, p. 10 |
| Q1 | L02, p. 23 |

---

## Coverage analysis: course materials vs. exam

This section compares two surfaces: **(a)** what the course materials cover, and
**(b)** what the decoded examples (the 7 photographed tickets + `Simulare.pdf`)
cover — and how they overlap.

### (a) What the course materials cover

The syllabus (`1_Prezentare curs/Content_of_the_AI_course_2025.pdf`) has exactly
three technical chapters, and the directory structure mirrors them:

**Ch. 1 — The Nature of Machine Learning** (`2_Invatare supervizata/`)
- 1.1 General model of learning from examples → inductive inference, how to
  analyze learning, **validity conditions for the ERM principle**
- 1.2 **Bias–variance tradeoff** (over-fitting)
- 1.3 **Regularization methods** (penalization / regularization terms)
- 1.4 Evaluation of performance → a posteriori empirical evaluation, model
  selection, **estimation of the real risk**, **tuning via a validation set**
- (the terms list also names: Bayesian decision principle, confusion matrix,
  precision-recall, confidence intervals, cross-validation, bagging)

**Ch. 2 — Supervised Learning of a Perceptron** (`3_Perceptronul/`)
- 2.1 Error-correction rules → **Perceptron (Rosenblatt) learning rule**,
  generalizations, **perceptron criterion function**, **May's rule**,
  **Widrow–Hoff rule**
- 2.2 The **Delta rules**
- 2.3 Other criterion functions
- 2.4 **Computation capabilities of a perceptron** (Boolean functions, linear
  separability)

**Ch. 3 — Learning Rule for Multilayer Feedforward NN** (`4_Rețele de perceptroni/`)
- 3.1 **Error backpropagation** (global-descent based)
- 3.2 Backprop enhancements → **weights initialization, learning rate, momentum,
  activation function**; also **Levenberg–Marquardt (LMBP)**
- 3.3 **Computation capabilities of an MFN**

**Applications** (`AA_1…AA_5`, `AA6 Matlab`): perceptron, Widrow-Hoff exercises,
MLP, backprop improvements, solved MLP problems, Matlab shallow-NN tooling.

**Non-examinable context** (`1_Prezentare curs`): AI history, the four definitions
of AI (think/act × humanly/rationally), prerequisites (linear algebra,
probability, numerical computation), curricula, research-lab/company context,
popular-science books. **Project** (`5_Proiect`): the Kaggle competition (the 30 %
component, evaluated separately by oral presentation).

### (b) What the decoded examples cover

The 18 distinct question types (A–R) from the master list above — all of which
fall inside chapters 1–3.

### Overlap map

| Course section | Exam question types hitting it | Tested? |
|---|---|---|
| 1.1.3 ERM validity / consistency | A (consistency), B (i.i.d.) | ✅ strong (worked solution) |
| 1.1 loss / intrinsic error | J (total error, noise-free) | ✅ |
| 1.2 Bias–variance / over-fitting | G (early stopping), M (risk vs complexity) | ✅ |
| 1.4.3 Estimation of real risk | M (2 estimation methods) | ✅ |
| 1.4.4 Validation-set tuning | G (train/valid + early stopping) | ✅ |
| **1.3 Regularization** | — | ❌ not seen |
| **1.4 Evaluation tooling** (confusion matrix, precision-recall, CV, confidence intervals, Bayesian) | — | ❌ not seen |
| 2.1.1 Perceptron / Rosenblatt rule | D (components + rule), F (definition) | ✅ strong |
| 2.4 Computation capabilities | F (which sets), I (AND), P (NOT) | ✅ strong |
| Perceptron convergence bound | H (K₀ max-epochs, hard points) | ✅ |
| **2.1.4 May's rule** | — | ❌ not seen |
| **2.1.5 Widrow–Hoff / 2.2 Delta rule** | — | ❌ not seen (only in AA_2 exercises) |
| **2.1.3 / 2.3 criterion functions** | — | ❌ not seen |
| 3.1 Backpropagation | C (chain net), E′ (batch), N (∂J/∂w₁), R (parallel net) | ✅ strong (worked solution) |
| 3.2 Backprop enhancements | K (2 speed-ups), L (saturation), Q (flat-spot / transfer fn) | ✅ |
| 3.3 MFN capabilities | E (f(x)=x feasibility) | ✅ |
| **3.2 LMBP / Levenberg–Marquardt** | — | ❌ not seen |

### Key conclusions

1. **The exam is a narrow technical slice of a much broader course.** Essentially
   the entire `1_Prezentare curs` half (history, philosophy, AI definitions,
   prerequisites, societal/industry context, popular books) and the project are
   **absent** from every decoded ticket. The written exam = chapters 1–3 only,
   i.e. the contents of `2_Invatare supervizata` plus the `Sinteza perceptron` and
   `Sinteza MLP` documents.

2. **Every decoded exam question maps to a course source — nothing is
   out-of-syllabus.** Mastering chapters 1–3 (especially the two "Sinteza" PDFs,
   which the Simulare grading scheme cites directly) covers 100 % of the observed
   exam content.

3. **Coverage is uneven inside chapters 1–3.** Heavily / repeatedly tested: ERM
   consistency, i.i.d., perceptron definition / capabilities (AND/NOT), the K₀
   convergence bound, the backprop derivation, and backprop speed-ups / saturation.
   **Present in the materials but never seen in the decoded examples** (potential
   blind spots, in rough priority): **regularization methods (1.3)**;
   **Widrow–Hoff / Delta rule** (despite a whole exercise folder, AA_2);
   **evaluation tooling** (confusion matrix, precision-recall, cross-validation,
   confidence intervals, Bayesian decision); **May's rule**; and
   **Levenberg–Marquardt (LMBP)**.

4. **Caveat.** The "not seen" column reflects only the 7 tickets + Simulare decoded
   here, not the full ticket pool. Since each ticket is meant to "cover the whole
   syllabus" and these topics *are* in the materials, they remain fair game — treat
   them as high-value review targets precisely because we have no worked example of
   them.

---

## Source files

| File | Content |
|-------|---------|
| `PHOTO-2026-05-26-20-35-55.jpg`   | Clean screenshot of the on-screen questions (Q1–Q4 visible; Q3 network diagram). |
| `PHOTO-2026-05-26-20-35-55-1.jpg` | Projector shot with red model solutions; reveals Q5 and answers to Q1–Q5. |
| `PHOTO-2026-05-26-20-35-55-2.jpg` | Wider projector shot of the same screen; reveals the full bottom derivations (Q3 backprop forward pass + J(a,b), tansig derivative; Q1 argmin definitions). |
| `PHOTO-2026-05-26-20-35-56.jpg`   | Near-duplicate of `...55-2.jpg` — same screen, same red solutions and bottom derivations. No new information. |
| `PHOTO-2026-05-26-20-35-56-1.jpg` | Handwritten page "B 12" — a second exam ticket (questions only). See the "Ticket B 12" section above. |
| `PHOTO-2026-05-26-20-35-56-2.jpg` | Typed printout "B18" — a third exam ticket (questions only). See the "Ticket B18" section above. |
| `PHOTO-2026-05-26-20-35-57.jpg`   | Handwritten page "Bilet 10" — a fourth exam ticket, 6 questions (questions only). See the "Bilet 10" section above. |
| `PHOTO-2026-05-26-20-35-57-1.jpg` | Handwritten page "Examen IA" — a fifth exam ticket, 5 questions (questions only). See the "Examen IA" section above. |
| `PHOTO-2026-05-26-20-35-57-2.jpg` | Typed printout "B11" — a sixth exam ticket (questions only). See the "Ticket B11" section above. |
| `PHOTO-2026-05-26-20-35-57-3.jpg` | Typed printout "B15" (label uncertain) — a seventh exam ticket (questions only). See the "Ticket B15" section above. |
| `Simulare.pdf`                    | The official mock exam — 5 questions with grading breakdowns and course-page references. See the "Simulare" section above. (`Simulare.docx` is the editable source of the same document.) |

> All photographs in this directory, plus `Simulare.pdf`, have now been decoded
> into this document.
