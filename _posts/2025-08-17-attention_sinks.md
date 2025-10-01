---
layout: distill
title: "Attention sinks from the graph perspective"
description: "Why causal transformers naturally concentrate attention on their earliest tokens."
tags: [transformers, attention, graph neural networks]
giscus_comments: true
date: 2025-08-17
featured: false
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Francesco Pappone
    url: "https://publish.obsidian.md/the-tensor-throne/Transformers+as+GNNs/Attention+sinks+from+the+graph+perspective"
    affiliations:
      name: "Università La Sapienza di Roma -- PSTP Technoscience"

toc:
  - name: Attention as message-passing
  - name: Causal Transformers and Attention Sinks
  - name: Wrapping up
  - name: Acknowledgements
  - name: Suggested citation

---

{% include figure.liquid
   loading="eager"
   path="assets/img/blog/attention_sinks/ChatGPT Image 24 ago 2025, 16_42_51.png"
   class="img-fluid rounded z-depth-1"
   zoomable=true
%}

Attention sinks have recently come back to the forefront of architecture discussion, especially due to their appearance in [gpt-oss](https://github.com/openai/gpt-oss) (although in a different form than the effect we're discussing today).

As a mechanism, attention sinks are easy to describe: when trained, decoder-only transformer models tend to allocate a disproportionate amount of attention to the first few tokens, and especially to the first.

This effect is well studied in its practical terms, and is often attributed to the model "offloading" probability mass to the early tokens to avoid their spurious allocation elsewhere. Recent works, like [Softpick](https://arxiv.org/abs/2504.20966), provide architectural choices that prevent sinks from forming. While this explanation may sound convincing at first glance, my intuition is still bothered by it: what do you mean the model "offloads"? Of course it doesn't explore that possibility intentionally, there must be some mechanism by which the attention sinks are either advantageous or a result of an intrinsic bias in the model. In this blogpost, we will argue that there is a significant bias in decoder-only transformers that may be to blame, at least partially, for this phenomenon. Moreover, this will also allow us to introduce a series of blogposts focused on analyzing transformers from the lens of message passing on graphs.

## Attention as message-passing

[Recent work by Chaitanya K. Joshi](https://arxiv.org/abs/2506.22084) has finally freed us from having to formalize independently a well known property of Transformers (and especially of attention layers): them being a special case of Graph Neural Networks (just like pretty much anything else, to be fair).

As a setting to our discussion, though, we will go over another angle with which attention can be seen as message-passing on a graph.

Most people are usually introduced to (multi-headed) self-attention directly via the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper. Despite this being generally a good practice in my opinion, it typically leads to attention being interpreted as the simplest way of making tokens interact in a transformer, or as just a soft version of a dictionary lookup. While neither view is wrong, such interpretations often drown out some interesting geometric details that lie in attention itself.

Let's start with regular, multi-headed attention.

Say you have $n$ tokens, with an embedding dimension $d$.

Let our input tokens be shaped as a matrix $X \in \mathbb{R}^{n \times d}$. We first process $X$ with three different linear projections, namely $W_q$, $W_k$ and $W_v$, and end up with the respective $Q \in \mathbb{R}^{n \times d_q}$, $K \in \mathbb{R}^{n \times d_k}$ and $V \in \mathbb{R}^{n \times d_v}$ matrices.

We then perform the well-known attention operation

$$
\mathrm{attention}(X) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Let's take a look at $\alpha = QK^\top$.
If we rewrite it component-wise we get

$$
\alpha_{ij} = \sum_{\ell=1}^{d_k} Q_{i\ell} K_{j\ell}.
$$

If we note that the rows of $Q$ and $K$ are respectively $q_i$ and $k_j$, we see that

$$
\alpha_{ij} = q_i k_j^\top = \langle q_i, k_j \rangle.
$$

The attention matrix $\alpha$'s entries are thus simply the Euclidean dot product between token embeddings, projected via the query and key matrices.

This still falls within the classical presentation of attention, so nothing to see here as of yet.

What if we could reinterpret these operations from a more geometric/topological perspective?
Let's take, for example,

$$
X \in \mathbb{R}^{\,n\times d}.
$$

And let's treat the rows of $X$ as a **point cloud**:

$$
X =
\begin{bmatrix}
x_1^\top\\
x_2^\top\\
\vdots\\
x_n^\top
\end{bmatrix},
\qquad
x_i \in \mathbb{R}^d.
$$

Constructing the $Q$, $K$, $V$ matrices for attention, we effectively project that cloud in three ways

$$
Q = X W_q \in \mathbb{R}^{\,n\times d_q},
$$

$$
K = X W_k \in \mathbb{R}^{\,n\times d_k},
$$

$$
V = X W_v \in \mathbb{R}^{\,n\times d_v}.
$$

We use these distinct projections to capture a **graph-like structure**, building an adjacency matrix between tokens, which can be seen as **nodes**

$$
\alpha_{ij} = \langle q_i, k_j \rangle,
\qquad
q_i = [Q]_{i,:},\; k_j = [K]_{j,:}.
$$

Stacking all scores:

$$
\alpha = Q K^\top \in \mathbb{R}^{\,n\times n}.
$$

The intuition is: the more points align in query-key space, the stronger their connection will be, and hence the stronger the link between the nodes.

Finally, we use softmax to normalize outgoing weights from each node

$$
A_{ij}
=
\frac{\exp\!\bigl(\alpha_{ij}/\sqrt{d_k}\bigr)}
{\displaystyle\sum_{j'=1}^n \exp\!\bigl(\alpha_{ij'}/\sqrt{d_k}\bigr)},
\qquad
A = \mathrm{softmax}\!\Bigl(\tfrac{\alpha}{\sqrt{d_k}}\Bigr).
$$

Each row of $A$ is a probability distribution and corresponds to the **node's neighbors**; small logits shrink toward 0, meaning most edge weights are very close to zero, apart from a few. This effectively heavily sparsifies the neighborhood, assigning most of the link weights to just a few connections, while the rest go to zero.

Lastly, the final operation

$$
\mathrm{attention}(X) = AV
$$

can now be interpreted from an interesting perspective: $V$ can be seen as a **vector-valued function defined on nodes of the graph**.

If we write it row-wise (hence focusing on each token, or node, at a time), we see that the updated function's value associated with the node becomes

$$
\mathrm{attention}(X)_i = \sum_\ell A_{i\ell} V_\ell.
$$

But what does multiplying a function defined on a graph by the adjacency mean? Let's say we have a directed graph $\mathcal{G} = (V,E)$ with adjacency $A$, with a function $f: v \rightarrow \mathbb{R}$ and $v \in V$.
Then, the multiplication $y = Af$ can be written, component-wise, as

$$
y_i = \sum_{j} A_{ij} f_j.
$$

Remember that, for an adjacency matrix, elements of column $i$ represent incoming links from other nodes in the graph. This means that $y_i$, or the result of the adjacency-multiplied function $f$, is the weighted average of $f$ over incoming nodes to node $i$, where the weights are decided by the adjacency matrix entries. Intuitively, you can think of this process as a sort of *diffusion*: features are aggregates of their neighbours. This means that, if we start with a rather unequally spatially distributed function (say a very localized highly positive region, and the rest being zero), then nodes on the boundary of the highly positive region would "diffuse" the highly positive values towards neighbouring nodes. Of course the topology of the graph heavily influences the speed of this diffusion. Unsurprisingly, this ties back very well with the actual physical phenomenon of heat diffusion, as we will see in a future blogpost.

## Causal Transformers and Attention Sinks

Note that the discussion so far has been agnostic of masking strategies applied to the attention score. While several uses of transformer models employ attention bidirectionally, LLMs, our large model protagonists, are usually causally masking attention to leverage parallelism for their next-token prediction task.

In our attention mechanism, this is done by substituting our $\alpha$ adjacency matrix with a masked, causal one, in the shape of $\alpha_m = \alpha \odot M$, with $M_{ij} = 1$ if $j \leq i$ and zero otherwise. Note that this gives our attention graph an even more interesting structure: our graph is now, by design, a **Directed Acyclic Graph** (DAG), meaning the graph contains no loops, and its adjacency matrix is nilpotent (meaning there exists $k$ such that $(A^k)_{ij} = 0$, $\forall i,j$).

One interesting corollary of this observation is that adjacency-based diffusion over DAGs is bound to accumulate information in sinks, specifically, in the first tokens of a causal model. This can be made explicit by looking at the shape of powers of $A$:

{% include figure.liquid
   path="assets/img/blog/attention_sinks/new_A^1.png"
   class="img-fluid rounded"
   zoomable=true
%}
{% include figure.liquid
   path="assets/img/blog/attention_sinks/new_A^2.png"
   class="img-fluid rounded"
   zoomable=true
%}
{% include figure.liquid
   path="assets/img/blog/attention_sinks/new_A^4.png"
   class="img-fluid rounded"
   zoomable=true
%}
{% include figure.liquid
   path="assets/img/blog/attention_sinks/new_A^8.png"
   class="img-fluid rounded"
   zoomable=true
%}

These plots (Fig. 1-4) show exactly what we expect on a DAG: as we take powers of the (masked) attention matrix $A$ the mass moves "leftward" toward early tokens. In the strictly lower-triangular case (no self-loops) this is a nilpotent operator, so sufficiently high powers collapse entirely into the earliest positions.

To connect this with learning dynamics, linearize one residual attention block (one head, for intuition; treat the MLP as a node-wise map) as

$$
X^{\ell+1} \approx X^{\ell} + A^{\ell} X^{\ell} B^{\ell},
\qquad
B^{\ell} = W_v^{\ell} W_o^{\ell}.
$$

Stacking $L$ such blocks yields an end-to-end map that is a polynomial in the $A^{\ell}$'s:

$$
X^{L} \approx \Big(\prod_{\ell=1}^{L} (I + A^{\ell} B^{\ell})\Big) X^{0}
= X^{0} + \sum_{\ell} A^{\ell} B^{\ell} X^{0}
+ \sum_{\ell_2 > \ell_1} A^{\ell_2} B^{\ell_2} A^{\ell_1} B^{\ell_1} X^{0} + \cdots
$$

When the $A^{\ell}$ are geometrically similar across depth, dominant terms behave like **powers of a causal $A$**. That is the same "multi-hop diffusion" we saw in the previous figures, progressively concentrating influence onto the first columns (early tokens).

But if that's the case during a forward pass, what makes a model exhibit this bias across training, as it's been noticed in the literature?

As it turns out, backprop itself mirrors this geometry. Gradients w.r.t. hidden states propagate with Jacobian transposes along the value path:

$$
g^{\ell} \approx (I + {B^{\ell+1}}^{\!\top} {A^{\ell+1}}^{\!\top}) \cdots (I + {B^{L}}^{\!\top} {A^{L}}^{\!\top}) g^{L}.
$$

Hence token-wise gradients accumulate along **column sums of products of $A$** (or, equivalently, row sums of products of $A^{\top}$). In a causal DAG those column sums are largest for earlier positions, so both activations **and** gradients preferentially route through (and update) paths that point to early tokens.

Practically, residual connections make the map a **polynomial** (not a single $A^k$), multi-head mixing and $B^{\ell}$ projections reshape directions, and layer-norm rescales signals. But the structural bias remains: deeper layers inherit updates that look like compositions of attention-diffusion steps, which, under causal masking, tend to be more and more "first-column concentrated".

Another corollary of our observation is that it would suggest that later layers are more subject to the attention sink phenomenon, while the very first layer should be much less impacted. This turns out to be true and well known when studying attention sinks, as is the case, for example, for [Llama 2](https://arxiv.org/abs/2309.17453), or in [this paper](https://openreview.net/pdf/736acc55a9b7a936dff081c2ba066c205279a844.pdf) and [this one](https://arxiv.org/pdf/2402.09221).

{% include figure.liquid
   path="assets/img/blog/attention_sinks/attention_sinks_in_llama.png"
   class="img-fluid rounded"
   zoomable=true
%}

Note that, while this **may not be the single effect responsible for attention sinks**, this means we should expect any causal decoder-only transformer to exhibit a bias towards allocating attention to its first few tokens (and increasingly so to the first).

This fundamentally clashes with many interpretations of sinks: several works characterize them as a useful feature that is learned by the model. If what we propose is true, it's exactly the opposite: when sinks **don't** show up, it means **the message-passing mechanism of your transformer is fundamentally flawed**, and hence it performs worse.

The attention sinks become a signal of **healthy communication** of tokens in attention, being a bias that is **intrinsic to the causal, decoder-only transformer**.

## Wrapping up

So, to recap, what does this mean? We identified a possible mechanism that may bias causal transformers to accumulate attention on their first few tokens. Note that we showed the mechanism in a highly simplified setting, and are proposing the idea that, despite those simplifications, the underlying effect is still strong enough to accumulate across training steps of a large transformer, and eventually explain the existence of attention sinks as we know them. In the next blogposts, we will use the same graph-centric framing of attention to analyze the problem of long context in transformer models, connecting it to heat diffusion and the oversmoothing and oversquashing phenomena known in the GNN literature. Stay tuned!

## Acknowledgements

Thanks a lot to [thelakosiffers](https://x.com/thelakosiffers), [Niccolò](https://x.com/Niccolg92), [Fabrizio](https://x.com/fabmilo), [Cynde](https://x.com/Cyndesama), [Francesco](https://x.com/f14bertolotti) and [Zed](https://x.com/zmkzmkz) for their precious feedback!

---

## Suggested citation

```bibtex
@misc{pappone2025attentionsinks,
  author       = {Francesco Pappone},
  title        = {Attention sinks from the graph perspective},
  year         = {2025},
  month        = {August},
  day          = {24},
  institution  = {Università La Sapienza di Roma -- PSTP Technoscience},
  howpublished = {\url{https://publish.obsidian.md/the-tensor-throne/Transformers+as+GNNs/Attention+sinks+from+the+graph+perspective}},
  note         = {Blogpost}
}
```
