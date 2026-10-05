# XGBoost: A Scalable Tree Boosting System

**Paper:** *XGBoost: A Scalable Tree Boosting System*
**Authors:** Tianqi Chen, Carlos Guestrin
**Published:** KDD 2016
**Area:** Machine Learning, Gradient Boosting, Decision Trees, Scalable ML Systems

---

## 📄 PAPER

Before reading this paper, I already knew how to use XGBoost in machine learning projects.

I knew that XGBoost is a powerful tree-based model and I had used it for problems like classification. I also knew some of its parameters such as:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `reg_alpha`
* `reg_lambda`

But I realized that **knowing how to use a model and understanding why it was designed that way are two different things.**

So I decided to read the original XGBoost paper.

My goal here is **not to reproduce the paper or implement XGBoost from scratch**.

Instead, I am trying to understand:

> **What problem were the authors trying to solve, what ideas did they introduce, and why does XGBoost work the way it does?**

---

# 🔎 My Reading Path

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
📊 DIAGRAM
   ↓
⚙️ ALGORITHM
   ↓
🔗 PRACTICAL CONNECTION
   ↓
🤔 MY QUESTIONS
   ↓
🧠 WHAT I LEARNED
```

---

# ❓ PROBLEM

## What problem is this paper actually solving?

The paper is not simply saying:

> "Let's create another boosting algorithm."

The bigger problem is:

> **How can we make tree boosting accurate while also making it fast and scalable for large datasets?**

Gradient boosting was already a strong machine learning technique.

But when the dataset becomes large, several problems appear.

For example:

* finding the best tree split can become expensive
* repeatedly sorting feature values can take time
* datasets can contain many missing or zero values
* storing and processing large datasets becomes difficult
* training can become slow
* distributed or out-of-memory training becomes challenging

So the authors were looking at the problem from two sides:

```text
                 TREE BOOSTING
                      │
          ┌───────────┴───────────┐
          ↓                       ↓
     ML / Algorithm          System / Hardware
          │                       │
          ↓                       ↓
   How to find good       How to process data
   tree splits?           efficiently?
          │                       │
          └───────────┬───────────┘
                      ↓
              SCALABLE XGBOOST
```

This was one of the things I found interesting in the paper.

**XGBoost is not only about the learning algorithm.**

A big part of the paper is also about **how the algorithm is implemented efficiently.**

---

# 💡 KEY IDEA

After reading the paper, I would summarize the main idea like this:

> **XGBoost combines a regularized gradient boosting objective with smarter tree-learning algorithms and system-level optimizations so that tree boosting can work efficiently on large and sparse datasets.**

I initially thought:

```text
XGBoost = Gradient Boosting + some improvements
```

But after reading the paper, I see it more like:

```text
                     XGBoost
                        │
       ┌────────────────┼────────────────┐
       ↓                ↓                ↓
  Optimization      Tree Learning     System Design
       │                │                │
       ↓                ↓                ↓
 Regularization    Better split      Column blocks
 Second-order      finding           Parallelism
 approximation     Sparse data       Cache efficiency
                                      Out-of-core
                                      Distributed
```

So there isn't just **one magic trick**.

Several ideas work together.

---

# 🧠 SIMPLE EXPLANATION

## First: What is boosting doing?

Suppose I have a model that is making mistakes.

Instead of trying to build one huge perfect tree, boosting builds trees **one after another**.

Each new tree tries to improve what the previous trees were doing.

A simplified view is:

```text
Training Data
      ↓
Initial Prediction
      ↓
Where is the model making mistakes?
      ↓
Build Tree 1
      ↓
Update Prediction
      ↓
What mistakes are still left?
      ↓
Build Tree 2
      ↓
Update Prediction
      ↓
Build Tree 3
      ↓
      ...
      ↓
Final Prediction
```

So the model gradually improves.

---

## But how does XGBoost decide what the next tree should do?

This is where the mathematics comes in.

Instead of simply looking at the raw prediction error, XGBoost looks at the **loss function** and uses information from its:

* first derivative → gradient
* second derivative → Hessian

This gives XGBoost more information about how the loss is changing.

---

# 📐 MATHEMATICS

## 1. The Objective Function

One of the important equations in the paper is:

\sum_i l(\hat y_i,y_i)
+
\sum_k \Omega(f_k)\

$$

At first this equation looks complicated.

But I can read it as:

```text
                XGBoost Objective
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
       Prediction Error      Model Complexity
             │                   │
             ↓                   ↓
            Loss          Regularization
             │                   │
             └─────────┬─────────┘
                       ↓
                  Final Objective
```

In simple words:

> **XGBoost does not only want predictions to be accurate. It also wants to control the complexity of the trees.**

The paper defines the tree complexity as:

\gamma T\
+\
\frac{1}{2}\lambda ||w||^2\
$$

where:

* $T$ = number of leaves
* $w$ = leaf weights
* $\gamma$ = penalty related to adding leaves
* $\lambda$ = regularization on leaf weights

So the objective is basically:

> **Make prediction errors small, but don't make the tree unnecessarily complicated.**

---

# 🧠 Why does regularization matter?

Imagine two trees give almost the same prediction quality.

### Tree A

```text
        Root
       /    \
      /      \
    Leaf    Leaf
```

Only a few leaves.

### Tree B

```text
                 Root
               /      \
             /          \
           /              \
        many              many
       leaves             leaves
```

Tree B is more complicated.

If both perform similarly, we don't necessarily want the unnecessarily complicated tree.

So XGBoost includes tree complexity directly in its objective.

That was an important connection for me:

> **Regularization is not something added separately after training. It is part of the objective used while building the tree.**

---

# 📐 2. Gradient and Hessian

For the current prediction, XGBoost calculates:

$$\
g_i =\
\frac{\partial l(y_i,\hat y_i)}\
{\partial \hat y_i}\
$$

and

$$\
h_i =\
\frac{\partial^2 l(y_i,\hat y_i)}\
{\partial \hat y_i^2}\
$$

I understand them in a simpler way as:

```text
Gradient
   ↓
Which direction should the prediction move?

Hessian
   ↓
How does the loss curve behave around that point?
   ↓
How strongly should we make the update?
```

A simple analogy:

> **Gradient tells me which way to go. Hessian gives me information about the shape of the road.**

This allows XGBoost to use a **second-order approximation** of the loss.

---

# 📐 3. Second-Order Approximation

The paper then uses a second-order approximation:

\sum_{i=1}^{n}
\left[
g_i f_t(x_i)
+
\frac{1}{2}h_i f_t(x_i)^2
\right]
+
\Omega(f_t)\

$$$

I don't want to look at this equation only as mathematics.

I read it as:

```text
For the new tree:

       Gradient information
                +
       Curvature information
                +
       Tree complexity
                ↓
       Decide how the new tree
       should improve the model
```

This was one of the main mathematical ideas I wanted to understand from the paper.

---

# 📐 4. What happens inside a leaf?

Suppose some training examples end up in the same leaf.

For those examples, we calculate:

$$\
G = \sum\_{i \in I} g_i\
$$$

and

$$\
H = \sum\_{i \in I} h_i\
$$

The paper gives the optimal leaf weight as:

-\frac{G}{H+\lambda}\

$$

In my words:

> XGBoost uses the accumulated gradient and Hessian information of the examples inside a leaf to decide what value that leaf should output.

So instead of thinking:

```text
Leaf → random prediction
```

I think:

```text
Examples reaching leaf
        ↓
Collect gradient information
        ↓
Collect Hessian information
        ↓
Apply regularization
        ↓
Calculate best leaf weight
```

---

# 📐 5. How does XGBoost choose a split?

This is another important part.

Suppose we have:

```text
Feature: Age

18
21
25
31
40
52
```

We could try different split points:

```text
Age < 21
Age < 25
Age < 31
Age < 40
Age < 52
```

For every possible split, we want to know:

> **Does this split improve the objective enough?**

The paper gives a split score:

\gamma\
$$

where:

* $G_L, H_L$ → gradient and Hessian sums on the left
* $G_R, H_R$ → gradient and Hessian sums on the right
* $G,H$ → values before splitting
* $\lambda$ → regularization
* $\gamma$ → penalty for creating the split

I don't need to memorize the whole equation.

The important idea I take from it is:

> **A split is useful only if separating the data into two groups gives enough improvement to justify the additional complexity.**

---

# 📊 DIAGRAM — SPLIT FINDING

```text
                 Current Node
                     │
                     │
             Try possible splits
                     │
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
    Split A       Split B       Split C
       │             │             │
       ↓             ↓             ↓
 Calculate        Calculate      Calculate
   Gain             Gain           Gain
       │             │             │
       └─────────────┼─────────────┘
                     ↓
              Compare gains
                     ↓
              Best split
                     ↓
              Split the node
```

So the tree is not just randomly choosing a feature and threshold.

It evaluates how useful a split is.

---

# ⚙️ ALGORITHM — EXACT SPLIT FINDING

The paper describes an **exact greedy algorithm**.

The basic idea is:

```text
For every feature
      ↓
Sort feature values
      ↓
Try possible split points
      ↓
Calculate gradient/Hessian statistics
      ↓
Calculate split gain
      ↓
Keep the best split
```

In pseudocode-like form:

```text
For each feature:
    sort the feature values

    move from left to right:
        add examples to the left side
        remaining examples stay on right

        calculate gain

        remember the best gain

Choose the split with the highest gain
```

This works, but there is a problem.

---

# ❓ Why not just check every possible split?

Because the dataset can be huge.

Imagine:

```text
1,000,000 rows
×
100 features
```

Trying every possible split repeatedly can become expensive.

This leads to another idea in the paper.

---

# 💡 APPROXIMATE SPLIT FINDING

Instead of checking every possible value, XGBoost can generate a smaller number of candidate split points.

For example:

```text
All possible values

1  2  3  4  5  6  7  8  9  10
|  |  |  |  |  |  |  |  |  |
          ↓

Candidate split points

2        5        8
|--------|--------|
```

Instead of checking everything:

```text
10 possible positions
```

we may check:

```text
3 candidate positions
```

The idea is:

> **Spend less computation on split search while keeping the important information needed to find a good split.**

---

# 📊 EXACT VS APPROXIMATE

```text
                 Split Finding
                      │
             ┌────────┴────────┐
             ↓                 ↓
           Exact           Approximate
             │                 │
             ↓                 ↓
       Check many/all      Candidate
       split points        split points
             │                 │
             ↓                 ↓
        More expensive      Faster
             │                 │
             ↓                 ↓
        More detailed      Small loss of
                           precision
```

The paper shows that a reasonable approximation can achieve accuracy close to exact greedy split finding.

That trade-off is interesting to me:

> **We don't always need to examine everything if we can keep the important information.**

---

# 📐 WEIGHTED QUANTILE SKETCH

This was one of the more difficult parts for me initially.

The paper introduces a **Weighted Quantile Sketch**.

The basic problem is:

> If we are going to choose candidate split points using quantiles, how should we choose them when the training examples have different importance/weights?

The paper defines a weighted rank:

\frac{
\sum_{(x,h)\in\mathcal D_k,\ x<z} h
}{
\sum_{(x,h)\in\mathcal D_k}h
}\

$$

The important part for me is not memorizing the equation.

It is understanding that:

```text
Normal quantile
      ↓
Looks at positions of values

Weighted quantile
      ↓
Also considers the importance/weight
of those values
```

So XGBoost's approximation is not simply:

> "Take random bins."

There is mathematical reasoning behind how the candidate split points are selected.

---

# 📊 WEIGHTED QUANTILE — INTUITION

Imagine we have:

```text
Value       Weight

10            1
20            1
30            1
40           10
50            1
```

The value `40` has much more weight.

A normal quantile calculation treats each row equally.

A weighted quantile considers that `40` represents much more weight in the distribution.

So:

```text
Normal view
10 ── 20 ── 30 ── 40 ── 50

Weighted view
10 ─ 20 ─ 30 ───────── 40 ─ 50
                     ↑
                much more weight
```

This helps the approximate split-finding process preserve useful information.

---

# ⚙️ SPARSITY-AWARE LEARNING

Another part I found interesting was how XGBoost deals with sparse data.

Real datasets can contain lots of:

- missing values
- zeros
- sparse features

For example, after one-hot encoding:

```text
Feature A   Feature B   Feature C   Feature D

1              0           0           0
0              0           1           0
0              0           0           1
0              1           0           0
```

Most values are zero.

A naive algorithm could waste computation processing all of those entries.

XGBoost instead uses a **sparsity-aware split-finding algorithm**.

The important idea is:

> **Learn where missing values should go instead of forcing us to manually fill them first.**

---

# 📊 SPARSITY-AWARE IDEA

```text
                 Feature
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
     Non-missing             Missing
       values                 values
          │                   │
          ↓                   │
     Find best split          │
          │                   │
          └─────────┬─────────┘
                    ↓
             Try default left
                    │
                    ↓
              Calculate gain
                    │
                    ↓
             Try default right
                    │
                    ↓
              Calculate gain
                    │
                    ↓
             Choose better one
```

So the model learns a **default direction** for missing values.

This is a clever idea because it turns missingness from only being a preprocessing problem into something the tree-learning process can handle.

---

# ⚙️ SYSTEM DESIGN

This is the part that changed my understanding of XGBoost the most.

When I first thought about XGBoost, I mostly thought about:

```text
Gradient Boosting
+
Decision Trees
```

But the paper spends a lot of effort on making the whole system efficient.

One important idea is the use of **column blocks**.

Instead of repeatedly reorganizing the same data during training, the data can be stored in a form that makes repeated split finding more efficient.

Simplified:

```text
Original Data

Rows
 ↓
┌────────────────────────────┐
│ Feature 1 | Feature 2 | ...│
│ Feature 1 | Feature 2 | ...│
│ Feature 1 | Feature 2 | ...│
└────────────────────────────┘
             ↓
       Column blocks
             ↓
   Reuse during training
             ↓
      Faster processing
```

The paper also discusses ideas such as:

- parallel processing
- cache-aware computation
- out-of-core learning
- distributed learning

So the paper is really combining:

```text
Algorithmic Ideas
        +
Systems Engineering
        =
Scalable Machine Learning
```

---

# 🧩 THE BIG PICTURE

After putting all the pieces together, this is how I currently understand XGBoost:

```text
                         XGBOOST
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ↓                 ↓                 ↓
     Optimization      Tree Learning      System Design
          │                 │                 │
          ↓                 ↓                 ↓
   Regularization     Exact splitting    Column blocks
   Gradients          Approximate         Parallelism
   Hessian            splitting           Cache efficiency
   2nd-order          Quantile sketch     Out-of-core
   approximation      Sparse handling     Distributed
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ↓
                     Scalable Boosting
```

This is probably the most useful diagram for me from the paper because it shows that **XGBoost is not one single technique**.

---

# 🔗 PRACTICAL CONNECTION

I have already used XGBoost in my machine learning projects.

For example, I have used it in a **customer churn prediction project**.

Before reading this paper, I mostly looked at XGBoost from the practical side:

```python
XGBClassifier(
    n_estimators=...,
    max_depth=...,
    learning_rate=...,
    reg_alpha=...,
    reg_lambda=...
)
```

I understood what these parameters did at a practical level.

But reading the paper helped me connect some of those parameters to the underlying ideas.

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
Controls tree complexity
    ↓
Can affect overfitting
```

I am not claiming that reading the paper suddenly makes me an expert in the internals of XGBoost.

What changed is that I now have a better mental model of **what is happening underneath the library I am already using.**

---

# 🤔 MY QUESTIONS WHILE READING

These are some questions I had while going through the paper:

### 1. Why does XGBoost need the Hessian?

I understood the gradient as the direction of change, but initially I wasn't comfortable with why the second derivative was useful.

My current understanding is that it provides information about the curvature of the loss and allows a second-order approximation.

---

### 2. Why put regularization directly into the objective?

I initially thought regularization was mainly a parameter we tune.

The paper helped me see that regularization is part of the optimization itself.

---

### 3. Why can't we simply check every possible split?

Because as the number of rows and features grows, repeatedly checking all possible split points becomes expensive.

This is where approximate split finding becomes useful.

---

### 4. Why are quantiles useful?

Because instead of considering every possible value, we can select representative candidate split points.

This reduces computation.

---

### 5. Why are the quantiles weighted?

Because the optimization uses gradient/Hessian information, so treating every example equally is not always enough for the split approximation.

---

### 6. How can missing values be handled without manually imputing everything?

The sparsity-aware algorithm can learn a default direction for missing values during split finding.

---

### 7. Which improvements are algorithmic and which are system-level?

This was one of the most useful things I noticed.

Some improvements change **how the model learns**.

Others change **how efficiently the computer performs the learning**.

---

# 🧠 WHAT I LEARNED

The biggest thing I learned from this paper is that **a good machine learning system is not only about the model itself.**

Before reading the paper, I mainly saw XGBoost as:

> "A very powerful gradient boosting algorithm using decision trees."

After reading it, my understanding became:

> **XGBoost is a combination of optimization ideas, tree-learning techniques, data structures and system-level engineering designed to make gradient tree boosting both effective and scalable.**

The three things I want to remember are:

### 1. Accuracy + Complexity

The objective does not only care about prediction error.

```text
Good prediction
      +
Controlled complexity
      ↓
Better objective
```

### 2. Exact vs Approximate

Sometimes checking everything is too expensive.

```text
Check everything
      ↓
Accurate but expensive

Use good candidate points
      ↓
Much cheaper
```

The important question becomes:

> **How much information can I remove without losing too much useful information?**

### 3. Algorithm + System

This was probably my biggest takeaway.

```text
Good ML idea
      +
Efficient implementation
      +
Good data handling
      ↓
Useful ML system
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
I understand why it uses
regularization,
gradients + Hessians,
split gains,
approximate split finding,
weighted quantiles,
sparsity-aware learning,
and system optimizations.
```

For me, this is what reading research papers is about:

> **Not just reading difficult equations, but trying to understand why those equations and ideas were needed in the first place.**

---

# 📚 PAPER REFERENCE

**Chen, T., & Guestrin, C. (2016).**\
*XGBoost: A Scalable Tree Boosting System.*\
Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

Original paper: [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)

---

## ⚠️ Note

This is a **research-paper reading and learning note**.

I am not claiming to have:

- reproduced the experiments from the paper
- implemented XGBoost from scratch
- reproduced the reported speedups
- contributed new research to XGBoost

The purpose of this page is to document how I read the paper, simplify the ideas in my own words, connect them with concepts I already know, and record what I learned.

