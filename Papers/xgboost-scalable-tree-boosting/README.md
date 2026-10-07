# XGBoost: A Scalable Tree Boosting System

**Paper:** *XGBoost: A Scalable Tree Boosting System*
**Authors:** Tianqi Chen, Carlos Guestrin
**Published:** KDD 2016
**Area:** Machine Learning, Gradient Boosting, Decision Trees, Scalable ML Systems

---

# 📄 PAPER

Before reading this paper, I already knew how to use XGBoost in machine learning projects.

I knew that XGBoost is a powerful tree-based model and I had used it for classification problems. I also knew parameters such as:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `reg_alpha`
* `reg_lambda`

But I realized that **knowing how to use a model and understanding why it was designed that way are two different things.**

So I decided to read the original XGBoost paper.

My goal here is **not to reproduce the paper or implement XGBoost from scratch**.

I want to understand:

> **What problem were the authors trying to solve, what mathematical ideas did they use, and how did those ideas become a scalable machine learning system?**

---

# 🔎 MY READING PATH

```text
📄 PAPER
   ↓
❓ PROBLEM
   ↓
💡 KEY IDEA
   ↓
🧠 SIMPLE EXPLANATION
   ↓
📐 MATHEMATICS
   ↓
📊 ALGORITHM
   ↓
⚙️ SYSTEM DESIGN
   ↓
🔗 PRACTICAL CONNECTION
   ↓
🤔 MY QUESTIONS
   ↓
🧠 WHAT I LEARNED
```

---

# ❓ PROBLEM

## What problem is the paper solving?

The paper is not simply proposing another tree-boosting algorithm.

The larger question is:

> **How can tree boosting remain accurate while also becoming computationally efficient and scalable to large datasets?**

Gradient tree boosting was already a powerful machine learning approach, but large-scale training creates several challenges:

* finding good tree splits can be expensive
* repeatedly processing feature values can be costly
* datasets can be sparse
* datasets may not fit completely in memory
* training needs to be efficient on multiple machines

The authors therefore approach the problem from both the **learning algorithm** and the **systems** side.

```text
                    TREE BOOSTING
                         │
          ┌──────────────┴──────────────┐
          ↓                             ↓
    Learning Algorithm             System Design
          │                             │
          ↓                             ↓
  How should the tree          How should the data
  be constructed?              be processed efficiently?
          │                             │
          └──────────────┬──────────────┘
                         ↓
                  SCALABLE BOOSTING
```

This was one of the first things I noticed:

> **XGBoost is not only about improving the prediction algorithm. It is also about engineering the training system efficiently.**

---

# 💡 KEY IDEA

My current understanding of the paper is:

> **XGBoost combines a regularized tree-boosting objective, second-order optimization, efficient split-finding methods, sparse-data handling, and system-level optimizations to make gradient tree boosting scalable.**

I initially thought:

```text
XGBoost
   =
Gradient Boosting
+
Some improvements
```

After reading the paper, I see it more like:

```text
                         XGBoost
                            │
          ┌─────────────────┼─────────────────┐
          ↓                 ↓                 ↓
     Optimization      Tree Learning      System Design
          │                 │                 │
          ↓                 ↓                 ↓
   Regularization     Split finding      Column blocks
   Gradient           Exact /            Parallelism
   Hessian            approximate         Cache awareness
   2nd-order          methods             Out-of-core
   approximation      Sparsity            Distributed
```

So there is not one single idea responsible for XGBoost.

Several ideas work together.

---

# 🧠 SIMPLE EXPLANATION

## Boosting

In boosting, trees are added sequentially.

A simplified view is:

```text
Training Data
      ↓
Initial Prediction
      ↓
Build a tree
      ↓
Update prediction
      ↓
Build another tree
      ↓
Update prediction
      ↓
Continue...
      ↓
Final Model
```

At each boosting round, the new tree is added to the existing model.

The prediction update can be written as:

### Equation

$$
\hat{y}_i^{(t)}
=
\hat{y}_i^{(t-1)}
+
f_t(x_i)
$$

Here:

* $\hat{y}_i^{(t)}$ = prediction after round $t$
* $\hat{y}_i^{(t-1)}$ = previous prediction
* $f_t$ = newly added tree
* $x_i$ = input example

So the new tree is not replacing the previous model.

It contributes an additional correction.

---

# 📐 MATHEMATICS

This is the part I found most useful because the paper connects the mathematical objective directly to how the tree is constructed.

---

## 1. Regularized Objective

The objective function used by XGBoost is:

$$
\mathcal{L}(\phi)
=
\sum_{i=1}^{n}
l(\hat{y}_i,y_i)
+
\sum_{k=1}^{K}
\Omega(f_k)
$$

**Equation (1)**

where:

* $l(\hat{y}_i,y_i)$ = training loss
* $f_k$ = the $k$-th tree
* $K$ = number of trees
* $\Omega(f_k)$ = complexity penalty for the tree

The regularization term is defined as:

$$
\Omega(f)
=
\gamma T
+
\frac{1}{2}\lambda
\|w\|^2
$$

**Equation (2)**

where:

* $T$ = number of leaves
* $w$ = vector of leaf weights
* $\gamma$ = penalty for tree complexity
* $\lambda$ = regularization on leaf weights

### My understanding

The objective has two parts:

```text
Prediction Loss
      +
Tree Complexity
      ↓
Overall Objective
```

So XGBoost is not only trying to minimize prediction error.

It is also controlling the complexity of the trees.

---

# 2. Objective at Boosting Round $t$

At boosting round $t$, the model already contains the previous trees.

The objective can be written as:

$$
\mathcal{L}^{(t)}
=
\sum_{i=1}^{n}
l
\left(
y_i,
\hat{y}_i^{(t-1)}
+
f_t(x_i)
\right)
+
\Omega(f_t)
$$

**Equation (3)**

The important part here is that we are now trying to find the **new tree $f_t$**.

So instead of optimizing all trees again, the current boosting step focuses on finding a good additional tree.

---

# 3. Gradient and Hessian

To approximate the loss for the new tree, XGBoost uses the first and second derivatives.

### Gradient

$$
g_i
=
\frac{
\partial l(y_i,\hat{y}_i^{(t-1)})
}{
\partial \hat{y}_i^{(t-1)}
}
$$

### Hessian

$$
h_i
=
\frac{
\partial^2 l(y_i,\hat{y}_i^{(t-1)})
}{
\partial (\hat{y}_i^{(t-1)})^2
}
$$

Here:

* $g_i$ describes the first-order change
* $h_i$ describes the second-order curvature

My simpler understanding:

```text
Gradient
    ↓
Direction of change

Hessian
    ↓
Curvature of the loss
```

The Hessian was one of the concepts I had to think about more carefully because I was initially much more familiar with gradient-based explanations than second-order optimization.

---

# 4. Second-Order Approximation

Using the gradient and Hessian, the paper applies a second-order Taylor approximation.

The objective becomes:

$$
\tilde{\mathcal{L}}^{(t)}
=
\sum_{i=1}^{n}
\left[
g_i f_t(x_i)
+
\frac{1}{2}
h_i f_t^2(x_i)
\right]
+
\Omega(f_t)
$$

**Equation (5)**

This equation was important for me because it connects the derivatives directly to tree construction.

The new tree is being evaluated using:

```text
Gradient information
        +
Hessian information
        +
Tree regularization
        ↓
New tree objective
```

So the mathematics is not separate from the tree-building process.

It is what allows the algorithm to evaluate candidate tree structures.

---

# 5. Representing a Tree

A tree can be represented using:

$$
f_t(x)
=
w_{q(x)}
$$

where:

* $q(x)$ assigns an example to a leaf
* $w_{q(x)}$ is the weight of that leaf

For a particular leaf $j$, let:

$$
I_j
=
\{i\mid q(x_i)=j\}
$$

This means $I_j$ contains the training examples that reach leaf $j$.

---

# 6. Gradient and Hessian Sums in a Leaf

For leaf $j$, define:

$$
G_j
=
\sum_{i\in I_j}g_i
$$

and

$$
H_j
=
\sum_{i\in I_j}h_i
$$

These two quantities summarize the examples that reach that leaf.

The optimization therefore does not need to treat every example independently when calculating the leaf weight.

It can work with the aggregated statistics $G_j$ and $H_j$.

---

# 7. Optimal Leaf Weight

For a fixed tree structure, the optimal weight of leaf $j$ is:

$$
w_j^*
=
-\frac{G_j}{H_j+\lambda}
$$

**Equation (5)**

This equation became much easier for me to understand after separating the notation:

```text
G_j
 ↓
Total gradient in the leaf

H_j
 ↓
Total Hessian in the leaf

λ
 ↓
Regularization

        ↓

Optimal leaf weight
```

So the leaf value is not chosen arbitrarily.

It is obtained by minimizing the regularized objective for that leaf.

---

# 8. Tree Structure and Objective

After substituting the optimal leaf weights, the objective for a tree can be written as:

$$
\tilde{\mathcal{L}}^{(t)}
=
-
\frac{1}{2}
\sum_{j=1}^{T}
\frac{G_j^2}{H_j+\lambda}
+
\gamma T
$$

**Equation (6)**

This equation is especially useful because it shows how the quality of a tree depends on:

* the gradient sums
* the Hessian sums
* regularization
* number of leaves

At this point, the question becomes:

> **How do we decide whether splitting a leaf makes the tree better?**

That leads to split finding.

---

# 9. Split Evaluation

Suppose a node is divided into a left child and a right child.

Define:

$$
G_L=\sum_{i\in I_L}g_i,
\qquad
H_L=\sum_{i\in I_L}h_i
$$

and:

$$
G_R=\sum_{i\in I_R}g_i,
\qquad
H_R=\sum_{i\in I_R}h_i
$$

Before splitting:

$$
G=G_L+G_R
$$

$$
H=H_L+H_R
$$

The paper evaluates the candidate split using:

$$
\mathcal{L}_{\text{split}}
=
\frac{1}{2}
\left[
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G^2}{H+\lambda}
\right]
-
\gamma
$$

**Equation (7)**

This is one of the most important equations for understanding how XGBoost chooses splits.

### My understanding

For every candidate split, the algorithm asks:

```text
Before split
     ↓
One leaf
     ↓
Evaluate objective

After split
     ↓
Two leaves
     ↓
Evaluate objective

Compare the two
     ↓
Is the split useful enough?
```

The $\gamma$ term matters because creating additional leaves increases model complexity.

So a split must provide enough improvement to justify that additional complexity.

---

# 📊 SPLIT FINDING

```text
                    Current Node
                         │
                         ↓
                Candidate split
                         │
          ┌──────────────┴──────────────┐
          ↓                             ↓
       Left child                    Right child
          │                             │
          ↓                             ↓
    Calculate G_L,H_L             Calculate G_R,H_R
          │                             │
          └──────────────┬──────────────┘
                         ↓
                  Evaluate split
                         ↓
                  Compare candidates
                         ↓
                   Best split
```

This is the connection I was looking for between the equation and the actual tree-building process.

---

# ⚙️ 10. Exact Greedy Split Finding

The exact greedy method considers candidate split points directly.

A simplified version is:

```text
For each feature
      ↓
Sort feature values
      ↓
Move through possible split points
      ↓
Update G_L and H_L
      ↓
Calculate G_R and H_R
      ↓
Evaluate split
      ↓
Keep the best candidate
```

For a sorted feature:

$$
x_{1k}\leq x_{2k}\leq\cdots\leq x_{nk}
$$

As the split point moves, the statistics can be updated:

$$
G_L
\leftarrow
G_L+g_j
$$

$$
H_L
\leftarrow
H_L+h_j
$$

and the right-side statistics can be obtained from the totals:

$$
G_R=G-G_L
$$

$$
H_R=H-H_L
$$

This avoids recomputing the sums from scratch for every candidate.

---

# ❓ Why approximate split finding?

The exact method can become expensive when the dataset contains a very large number of rows and features.

Instead of examining every possible split point, XGBoost can use a smaller set of candidate thresholds.

```text
Exact
────────────────────────────
Check many possible points

Approximate
────────────────────────────
Select representative points
```

The goal is to reduce computation while keeping enough information to find good splits.

---

# 📐 11. Weighted Quantile Sketch

For approximate split finding, the paper introduces the **Weighted Quantile Sketch**.

For feature $k$, the weighted rank function is:

$$
r_k(z)
=
\frac{
\displaystyle
\sum_{\substack{(x,h)\in\mathcal{D}_k\\x<z}}h
}{
\displaystyle
\sum_{(x,h)\in\mathcal{D}_k}h
}
$$

The important difference from an ordinary quantile is that the distribution is weighted using the Hessian values.

Candidate points are selected so that the weighted distribution is represented with controlled error.

The paper expresses the approximation condition as:

$$
r_k(s_{k,j+1})
-
r_k(s_{k,j})
<
\epsilon
$$

where $\epsilon$ controls the approximation accuracy.

### My understanding

I do not want to memorize this equation.

What I understand is:

```text
All possible feature values
          ↓
Weighted distribution
          ↓
Select representative points
          ↓
Use them as candidate splits
          ↓
Reduce split-search cost
```

The weighting is important because the optimization itself uses Hessian information.

---

# ⚙️ 12. Sparsity-Aware Learning

Large real-world datasets can contain many missing or zero entries.

This is especially common with sparse features such as one-hot encoded variables.

A simplified example:

```text
Feature A   Feature B   Feature C   Feature D

1              0           0           0
0              0           1           0
0              0           0           1
0              1           0           0
```

XGBoost introduces a sparsity-aware split-finding method.

The important idea is that the algorithm can learn a **default direction** for missing values while evaluating a split.

```text
                 Candidate Split
                       │
              ┌────────┴────────┐
              ↓                 ↓
        Default Left       Default Right
              │                 │
              ↓                 ↓
        Evaluate score     Evaluate score
              │                 │
              └────────┬────────┘
                       ↓
                 Better direction
```

This means missing values do not necessarily have to be handled only through a separate preprocessing step.

---

# ⚙️ 13. System-Level Design

This was the part that changed my understanding of XGBoost the most.

The paper does not stop at the mathematical optimization.

It also discusses how to make the implementation efficient.

Important system ideas include:

* column blocks
* parallel computation
* cache-aware computation
* out-of-core learning
* distributed learning

A simplified picture is:

```text
                 XGBoost
                    │
       ┌────────────┴────────────┐
       ↓                         ↓
  Learning Algorithm        System Design
       │                         │
       ↓                         ↓
  Gradient/Hessian          Data layout
  Split finding             Parallelism
  Regularization            Cache usage
  Sparse handling           Out-of-core
                             Distributed
```

One important idea is organizing data into **column blocks** so that the same data organization can be reused during split finding.

This is where I started seeing XGBoost as more than just a machine learning formula.

It is also a carefully engineered training system.

---

# 🧩 THE BIG PICTURE

After putting the paper together, this is my current mental model:

```text
                         XGBoost
                            │
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
   Optimization        Tree Learning       System Design
        │                   │                   │
        ↓                   ↓                   ↓
  Regularization       Split finding      Column blocks
  Gradient             Exact method       Parallelism
  Hessian              Approximation      Cache efficiency
  2nd-order            Quantile sketch    Out-of-core
  approximation        Sparse handling    Distributed
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ↓
                  Scalable Tree Boosting
```

This is probably my biggest understanding from the paper:

> **XGBoost is not one single improvement. It is a combination of mathematical optimization, tree-learning methods, and systems engineering.**

---

# 🔗 PRACTICAL CONNECTION

I have already used XGBoost in my machine learning projects.

For example, I have used it in a **customer churn prediction project**.

Before reading the paper, I mostly looked at XGBoost from the practical side:

```python
XGBClassifier(
    n_estimators=...,
    max_depth=...,
    learning_rate=...,
    reg_alpha=...,
    reg_lambda=...
)
```

I understood these parameters at a practical level.

Reading the paper helped me connect some of them with the underlying ideas.

For example:

```text
reg_lambda
      ↓
Regularization
      ↓
Penalty on leaf weights
```

and:

```text
max_depth
      ↓
Limits tree depth
      ↓
Controls part of the tree complexity
```

I am not claiming that reading this paper makes me an expert in the internals of XGBoost.

What changed is that I now have a better mental model of **what is happening underneath the library I was already using.**

---

# 🤔 MY QUESTIONS WHILE READING

### 1. Why does XGBoost use the Hessian?

I understood the gradient more easily, but initially I was not comfortable with why the second derivative was useful.

My current understanding is that the Hessian provides curvature information and allows the objective to be approximated using second-order information.

---

### 2. Why is regularization part of the objective?

I initially thought of regularization mainly as a parameter that I tune.

The paper helped me understand that the complexity penalty is included directly in the objective being optimized.

---

### 3. Why not check every possible split?

Because the number of possible split points becomes expensive to evaluate as the dataset grows.

This motivates approximate split finding.

---

### 4. Why are quantiles useful?

They provide a way to select representative candidate split points instead of checking every possible feature value.

---

### 5. Why are the quantiles weighted?

Because the approximate split-finding method uses Hessian-based weights when constructing the candidate distribution.

---

### 6. How are missing values handled?

The sparsity-aware algorithm can learn a default direction for missing values during split evaluation.

---

### 7. What is the difference between an algorithmic improvement and a system improvement?

This was one of the most useful distinctions for me.

Some ideas improve **how the model learns**.

Others improve **how efficiently the computer performs the learning**.

---

# 🧠 WHAT I LEARNED

The biggest thing I learned from this paper is that **a good machine learning system is not only about the model itself.**

Before reading the paper, I mainly saw XGBoost as:

> "A powerful gradient boosting algorithm using decision trees."

After reading it, my understanding became:

> **XGBoost combines optimization ideas, tree-learning techniques, sparse-data handling, and system-level engineering to make gradient tree boosting effective and scalable.**

The three things I want to remember are:

### 1. Accuracy + Complexity

The objective considers both prediction loss and model complexity.

```text
Prediction quality
        +
Controlled complexity
        ↓
Regularized objective
```

### 2. Exact vs Approximate

Checking every possible split can be expensive.

```text
Exact
  ↓
More candidate points
  ↓
More computation

Approximate
  ↓
Representative candidate points
  ↓
Less computation
```

The important question becomes:

> **How much computation can we save while keeping enough information to find useful splits?**

### 3. Algorithm + System

This was probably my biggest takeaway.

```text
Good ML idea
      +
Efficient algorithm
      +
Efficient implementation
      ↓
Scalable ML system
```

So when I see a machine learning library now, I want to understand not only:

> **"How do I use it?"**

but also:

> **"Why was it designed this way?"**

---

# 📝 MY FINAL TAKEAWAY

I started reading this paper because I already knew how to **use XGBoost**.

I wanted to understand what was happening underneath the library.

I am still learning the mathematical details, especially the second-order optimization and weighted quantile sketch, but I now have a clearer picture of the overall design.

The biggest change in my understanding is:

```text
Before

"I know XGBoost."
        ↓
I can train an XGBoost model.


After reading the paper

"I understand more about XGBoost."
        ↓
I understand the connection between
the objective,
gradients + Hessians,
leaf weights,
split evaluation,
approximate split finding,
sparsity handling,
and system optimizations.
```

For me, this is what reading research papers is about:

> **Not just reading difficult equations, but trying to understand why those equations and ideas were needed in the first place.**

---

# 📚 PAPER REFERENCE

**Chen, T., & Guestrin, C. (2016).**
*XGBoost: A Scalable Tree Boosting System.*
Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

Original paper: https://arxiv.org/abs/1603.02754

---

## ⚠️ NOTE

This is a **research-paper reading and learning note**.

I am not claiming to have:

* reproduced the experiments from the paper
* implemented XGBoost from scratch
* reproduced the reported speedups
* contributed new research to XGBoost

The purpose of this page is to document how I read the paper, simplify the ideas in my own words, connect them with concepts I already know, and record what I learned.
