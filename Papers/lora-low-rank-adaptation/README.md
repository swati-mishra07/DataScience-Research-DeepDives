# LoRA: Low-Rank Adaptation of Large Language Models

**Paper:** *LoRA: Low-Rank Adaptation of Large Language Models*
**Authors:** Edward J. Hu et al.
**Published:** ICLR 2022
**Area:** Large Language Models, Fine-Tuning, Parameter-Efficient Fine-Tuning (PEFT)

> **Note:** This is my learning-based breakdown of the paper, written in my own words. I am not implementing or reproducing the paper here.
>
> **Sections I read:** Abstract, Introduction, Section 2, Section 3, Section 4, and Conclusion.

---

# 📄 PAPER

The paper introduces **LoRA (Low-Rank Adaptation)** as a way to adapt a large pretrained model to a specific task without updating all of its original parameters.

The main idea sounds simple:

> **Keep the original model weights frozen and learn a much smaller update instead.**

What made this paper interesting to me is that the authors are not trying to make the pretrained model smaller. Instead, they are asking:

**Do we really need to change the whole model when adapting it to a new task?**

---

# ❓ PROBLEM

Large pretrained models contain a huge number of parameters.

For example, suppose we already have a pretrained model and want to fine-tune it for a downstream task.

With normal fine-tuning, we update the model's weights:

$$
W' = W + \Delta W
$$

where:

* \(W\) = original pretrained weights
* \(\Delta W\) = change learned during fine-tuning
* \(W'\) = updated weights

The problem is that \(\Delta W\) can be almost as large as \(W\).

So if the original model is very large, fine-tuning can require:

* a lot of GPU memory
* storing many trainable parameters
* storing gradients
* storing optimizer states
* separate copies of the model for different tasks

This becomes expensive when working with large language models.

---

# 💡 KEY IDEA

The key idea of LoRA is:

**Don't learn the entire weight update \(\Delta W\). Learn a smaller low-rank representation of it.**

Instead of directly learning:

$$
\Delta W
$$

LoRA represents it as:

$$
\Delta W = BA
$$

Therefore:

$$
\boxed{W' = W + BA}
$$

Here:

* \(W\) = frozen pretrained weight matrix
* \(A\) = trainable low-rank matrix
* \(B\) = trainable low-rank matrix
* \(BA\) = learned update to the original weights

The important part is that \(A\) and \(B\) are much smaller than the original matrix.

---

# 🧠 SIMPLE EXPLANATION

The easiest way I understood LoRA is this:

Imagine I have a huge pretrained model.

I don't want to rewrite the whole model for every new task.

So I keep the original model as it is and learn a **small additional change** that teaches the model the new task.

Something like:

```text
                 PRETRAINED MODEL
                       │
                       │
                 W (frozen)
                       │
                       │
              ┌────────┴────────┐
              │                 │
              │                 │
        Original W          LoRA update
                              BA
                               │
                               ▼
                         W + BA
                               │
                               ▼
                     Adapted Model
```

So LoRA is basically saying:

**"Keep what the model already knows. Learn only the change that is needed."**

That was the main idea I took from the paper.

---

# 📐 WHY "LOW-RANK"?

This was one of the parts I had to understand carefully.

Suppose the original weight matrix is:

$$
W \in \mathbb{R}^{d \times k}
$$

In normal fine-tuning, the update would also have the same size:

$$
\Delta W \in \mathbb{R}^{d \times k}
$$

LoRA instead uses two smaller matrices:

$$
A \in \mathbb{R}^{r \times k}
$$

and

$$
B \in \mathbb{R}^{d \times r}
$$

where:

$$
r \ll \min(d,k)
$$

Then:

$$
\Delta W = BA
$$

The resulting dimensions are:

$$
(d \times r)(r \times k) = d \times k
$$

So \(BA\) has the **same shape as \(W\)**, but we don't have to directly learn all \(d \times k\) values.

---

# 🔢 PARAMETER COMPARISON

Suppose:

$$
d = k = 10,000
$$

A full weight update would contain:

$$
10,000 \times 10,000 = 100,000,000
$$

parameters.

That is **100 million parameters** just for the update.

Now suppose LoRA uses:

$$
r = 8
$$

Then:

$$
A \in \mathbb{R}^{8 \times 10,000}
$$

and

$$
B \in \mathbb{R}^{10,000 \times 8}
$$

Number of trainable parameters:

$$
8(10,000) + (10,000)(8)
$$

$$
= 80,000 + 80,000
$$

$$
= 160,000
$$

So instead of learning:

$$
100,000,000
$$

parameters, LoRA learns:

$$
160,000
$$

parameters.

That's the main reason the low-rank representation is useful.

---

# 🤔 WHAT DOES "RANK" ACTUALLY MEAN?

Initially, I thought rank was some kind of ranking of the model.

It is not.

Here, **rank is a mathematical property of a matrix**.

In LoRA, \(r\) controls the size of the low-rank representation.

A smaller \(r\) means fewer trainable parameters.

A larger \(r\) means more trainable parameters and potentially more capacity for the adaptation.

So there is a trade-off:

```text
Smaller r
   ↓
Fewer parameters
   ↓
Less memory / computation
   ↓
But potentially less adaptation capacity


Larger r
   ↓
More parameters
   ↓
More adaptation capacity
   ↓
But more resources
```

The important thing is that \(r\) is chosen to be much smaller than the dimensions of the original matrix.

---

# 🧩 WHAT IF THE RANK IS TOO SMALL?

This was one of my questions while reading the paper.

If \(r\) is extremely small, the LoRA update has limited capacity.

It may not be able to represent a sufficiently useful update for a particular task.

So low rank does not mean:

**"The smaller the better."**

It means:

**"Can a much smaller update capture enough of the change needed for the task?"**

The paper's motivation comes partly from the observation that the updates produced by full fine-tuning can have a low effective rank.

---

# 🔍 RANK-DEFICIENCY

The paper discusses the idea of **rank-deficiency** in the updates produced by fine-tuning.

In simple terms:

A matrix can have a maximum possible rank based on its dimensions, but the actual learned update may have a much smaller rank.

For example, a matrix might theoretically support rank 100, but the important information in the learned update may effectively lie in a much smaller-dimensional space.

This observation motivated the authors to ask:

> Maybe we don't need to learn the entire update matrix.

Instead, perhaps a low-rank approximation can capture the important part of the adaptation.

This is one of the main ideas behind LoRA.

---

# ⚙️ HOW LoRA WORKS

The basic process can be understood in four steps.

### 1. Start with a pretrained model

We already have a pretrained model with weights \(W\).

### 2. Freeze the original weights

The original parameters are not updated during LoRA training.

$$
W = \text{frozen}
$$

### 3. Add trainable low-rank matrices

LoRA introduces:

$$
A \quad \text{and} \quad B
$$

and learns them during training.

The update is:

$$
\Delta W = BA
$$

### 4. Use the adapted weight

The effective weight becomes:

$$
\boxed{W' = W + BA}
$$

So the model gets task-specific adaptation without directly changing the original \(W\).

---

# 🧠 WHAT IS ACTUALLY BEING TRAINED?

This distinction is important.

In full fine-tuning:

```text
Pretrained Model
       ↓
Most/all model parameters
       ↓
Updated
```

With LoRA:

```text
Pretrained Model
       ↓
Original weights W
       ↓
Frozen ❄️

LoRA
 ↓
A and B
 ↓
Trainable 🔥
```

So the pretrained model is still there.

We are mainly learning the smaller LoRA parameters.

---

# 📊 WHY THIS CAN SAVE MEMORY

Training a neural network requires more than just storing the model weights.

During training, we may also need:

* gradients
* optimizer states
* activations
* trainable parameters

If most of the pretrained model is frozen, we don't need to maintain the same training state for all of those original parameters.

Therefore, LoRA can significantly reduce the resources needed for fine-tuning large models.

This is one of the main practical reasons I found LoRA interesting.

---

# 🚀 LoRA AND INFERENCE

There is another part of the paper that I initially mixed up with training efficiency.

**Training efficiency and inference efficiency are not exactly the same thing.**

During training, LoRA reduces the number of trainable parameters.

For inference, LoRA has another useful property.

After training, the learned update can be merged with the original weights:

$$
W' = W + BA
$$

Once merged, the model can use the resulting weight directly.

So there doesn't necessarily have to be a separate LoRA computation path during inference.

This helps avoid the inference latency overhead that some adapter-based approaches can introduce.

---

# 📊 WHAT I UNDERSTOOD FROM TABLE 1

The paper compares the inference latency of different approaches.

The experiments use:

* GPT-2 medium
* NVIDIA Quadro RTX8000
* 100 trials
* different batch sizes
* different sequence lengths

The table reports latency in milliseconds.

One thing that stood out to me was the difference in overhead for short sequences.

For example, with:

$$
\text{Batch Size}=1
$$

and:

$$
\text{Sequence Length}=128
$$

the reported latency overhead was:

* AdapterL: **+20.7%**
* AdapterH: **+30.3%**

The important point I took from this is:

**Adding separate adapter layers can introduce additional inference computation, and that overhead can become more noticeable in online scenarios with short sequences.**

LoRA's ability to merge its update into the original weights is useful here.

---

# 🔬 FULL FINE-TUNING VS LoRA

|                          | Full Fine-Tuning           | LoRA                                 |
| ------------------------ | -------------------------- | ------------------------------------ |
| Original weights         | Updated                    | Frozen                               |
| Trainable parameters     | Very large                 | Much smaller                         |
| Training memory          | Higher                     | Lower                                |
| Task-specific parameters | Large                      | Small                                |
| Multiple tasks           | Separate fine-tuned models | Base model + different LoRA adapters |
| Main idea                | Change the model           | Learn a smaller update               |

The important difference is not that LoRA creates a completely new model.

It creates a **small task-specific adaptation** to an existing pretrained model.

---

# 🧩 ONE BASE MODEL, MULTIPLE LoRA ADAPTERS

Another practical idea I took from this paper is how LoRA can be useful when one base model needs to support different tasks.

For example:

```text
                 Base Model
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
     LoRA A       LoRA B       LoRA C
        │            │            │
        ▼            ▼            ▼
    Task A        Task B        Task C
```

Instead of keeping a completely separate full model for every task, we can keep the common pretrained model and store smaller task-specific adaptations.

This makes the idea especially interesting for large models.

---

# 🔍 HOW LoRA IS DIFFERENT FROM OTHER METHODS

LoRA is not the only way to avoid full fine-tuning.

The paper compares its approach with other parameter-efficient methods such as:

* Adapter-based methods
* Prefix tuning
* Prompt tuning
* Other approaches for reducing the number of trainable parameters

What I found interesting is that the goal is similar, but the way each method introduces task-specific information is different.

LoRA does this through a **low-rank update to selected weight matrices**.

---

# 🤔 HOW DO WE CHOOSE WHERE TO APPLY LoRA?

This became one of my questions after reading the paper.

LoRA does not necessarily have to be applied to every weight matrix.

The authors discuss applying LoRA to selected matrices in the Transformer architecture.

This creates an important question:

**Which weight matrices should actually receive the LoRA update?**

The paper points out that the choice is largely based on practical experimentation rather than a completely principled rule.

That means there is still room for better ways of deciding where LoRA should be applied.

---

# 🤔 MY QUESTIONS

These are questions I had while reading the paper:

### 1. How small can the rank \(r\) become before performance starts dropping significantly?

I understand that a smaller rank reduces parameters, but there must be a point where the update becomes too limited.

### 2. Why do fine-tuning updates often have low-rank structure?

The paper uses this observation as motivation, but I want to understand the deeper reason behind it.

### 3. How should we choose the matrices where LoRA is applied?

Is there a general rule, or does it mostly depend on experiments and the task?

### 4. Does the best rank remain the same for different tasks?

I would like to understand whether different tasks require very different values of \(r\).

### 5. Why does a low-rank update work so well for some large models?

I understand the mathematical construction of \(BA\), but the deeper connection between the low-rank update and the knowledge learned by the model is still something I want to explore.

---

# 🚧 OPEN QUESTIONS FROM THE AUTHORS

The paper itself also points toward several areas that are not fully understood.

### 1. Combining LoRA with other efficient adaptation methods

The authors suggest that LoRA could potentially be combined with other parameter-efficient methods.

The interesting idea here is that different methods may improve different parts of the adaptation process.

So combining them could potentially provide additional benefits.

---

### 2. Understanding what happens during fine-tuning

The authors point out that the mechanism behind fine-tuning is still not completely understood.

One interesting question is:

**How does pretraining knowledge get transformed when the model is adapted to a downstream task?**

The authors suggest that studying LoRA updates may make this question more tractable.

Here, "more tractable" basically means **easier to study or analyze**, not necessarily easy.

---

### 3. Better ways to choose weight matrices

The choice of which matrices to apply LoRA to is largely heuristic.

A **heuristic** here means a practical rule that works reasonably well, but is not necessarily derived from a complete theoretical explanation.

So a better, more principled method for selecting weight matrices is still an open direction.

---

### 4. Is the original weight matrix also rank-deficient?

The paper observes that fine-tuning updates can have low-rank structure.

This raises another interesting question:

If the update is rank-deficient, could the original pretrained weight matrix \(W\) also contain some form of rank deficiency?

The paper raises this as a direction for future research.

It is important to note that this is a **question**, not a conclusion that the original model weights are definitely rank-deficient.

---

# 🔗 PRACTICAL / INDUSTRY CONNECTION

I have already seen LoRA in the context of fine-tuning language models, so this paper helped me understand what is happening behind the idea.

The practical picture I now have is:

```text
Large Pretrained Model
          │
          │ freeze
          ▼
     Base Model
          │
          ├──────── LoRA Adapter A → Task A
          │
          ├──────── LoRA Adapter B → Task B
          │
          └──────── LoRA Adapter C → Task C
```

Instead of modifying the entire large model for every task, we can keep the common base model and learn smaller task-specific adaptations.

This is especially useful when working with models that are expensive to fully fine-tune.

---

# 🧠 WHAT I LEARNED

Before reading this paper, I mostly thought of fine-tuning as:

> Take a pretrained model and train it again on a new dataset.

After reading LoRA, I understand that this does not necessarily mean updating the whole model.

The biggest things I learned are:

1. **LoRA is a fine-tuning/adaptation method, not a new model.**

2. **The original pretrained weights can remain frozen.**

3. **Instead of directly learning a huge \(\Delta W\), LoRA represents it as:**

$$
\Delta W = BA
$$

4. **The effective weight becomes:**

$$
W' = W + BA
$$

5. **The rank \(r\) controls the size of the low-rank adaptation.**

6. **A smaller rank means fewer trainable parameters, but potentially less adaptation capacity.**

7. **LoRA mainly helps make fine-tuning more parameter-efficient and memory-efficient.**

8. **Training efficiency and inference efficiency are different things.**

9. **LoRA can merge its learned update with the original weights, which can avoid additional inference overhead from a separate adapter path.**

10. **The paper also made me realize that we still don't completely understand why low-rank adaptation works so well.**

---

# 💭 MY BIGGEST TAKEAWAY

The biggest idea I took from this paper is:

> **Adapting a large model does not necessarily mean changing the whole model.**

Sometimes, the useful change needed for a new task can be represented by a much smaller structured update.

That is the part of LoRA that I found most interesting.

---

# 📚 REFERENCE

**Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W.**

*LoRA: Low-Rank Adaptation of Large Language Models.*

International Conference on Learning Representations (ICLR), 2022.

---

## 📌 My Reading Scope

For this deep dive, I read:

* Abstract
* Introduction
* Section 2
* Section 3
* Section 4
* Conclusion

I have **not** treated the sections I did not read as if I had studied them in detail.
