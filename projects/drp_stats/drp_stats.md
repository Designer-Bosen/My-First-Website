[← Back](../index.html)

# Stats 199 - Directed Research

---

## Navigation Links

[Tentative Research Proposal](index.html?file=drp_stats_proposal.md)

[Anchor Paper Ideas](index.html?file=drp_stats_anchor_paper_ideas.md)

[Weekly Meeting Summary](index.html?file=drp_stats_meeting_sum.md)

---

## Project Description

This **Stats 199 - Directed Research** is a 1-quarter contract course at UCLA Statistics and Data Science conducted under the supervision of Professor George Michailidis. The research is built upon on the recent work [High Dimensional Logistic Regression Under Network Dependence](https://jmlr.org/papers/volume25/22-1040/22-1040.pdf) which addresses the presence of network dependence in observations via introducing Ising-type network interaction structure into the logistic regression model. The paper proosed a penalized maximum pseudo-likelihood (PMPL) approach to overcome computational burden of full likelihood. This approach enables efficient estimation in high-dimensional setting while achieving descent computational rate and approximation accuracy. In particular, under sparsity and weak dependence conditions, PMPL estimator achieves consistency comparable to classical logistic regression with independent data.

Motivated by my research interest in high dimensional Bayesian inference...



---
## My Direction

Given above results from the anchor paper, I want to formulate a Bayesian logistic regression model under network dependence. Then analyze its properties under this regime e.g. posterior contraction rates, consistency, computational approaches such as MCMC.

Keywords: High-dimensional Bayesian Inference, restricted strong convexity

### Prior Knowledge on Bayesian Inference

**Bayesian Theorem:** $\mathbb{P}(A \mid B)=\frac{\mathbb{P}(B \mid A)\cdot \mathbb{P}(A)}{\mathbb{P}(B)}$

From Bayesian perspective, parameter $\theta$ is treated as a random variable and follows some distribution. If $Y$ represents the data, the probability density function of $\theta$ give $Y$ can be represented as:

$$f(\theta \mid Y) = \frac{f(Y \mid \theta) \cdot f(\theta)}{f(Y)} = \frac{\text{Likelihood} \cdot \text{Prior}}{\text{Normalizing Constant}} \propto \text{Likelihood} \cdot \text{Prior}$$

---

## Phase 1: Model the network in Python

**Adjacency Matrix** $A \in \mathbb{R}^{N\times N}_{sym}$ represent the network struture. If node $i$ is connected to node $j$, then $A_{ij} = A_{ji} = 1$. Otherwise $A_{ij} = A_{ji} = 0$. Each non-diagonal entry is generated using a Bernoulli distribution with tunable parameter $s$ controlling sparsity.

**Covariate Matrix** $Z \in \mathbb{R}^{N\times p}$ represent the collection of node-specific features, where $Z_i$ corresponds to the covariate vector of node $i$. $Z$ captures the individual effects, and entries of Z are generated independent from a chosen distribution.

**Outcome Vector** $X \in \{0,1\}^{N}$ where $X_i \in \{0, 1\}$ is generated through a iterative updating process inspired by Gibbs samppling.
For each node in each iteration, $X_i$ is mapped from a probablistic function $\sigma$ using the score vector $S$ and $S_i=Z_i\theta + \beta \cdot m_i=\text{Individual Feature} + \beta \times \text{Neighbor Influence}$ where individual effect $k_i = Z_i \cdot \theta$ is determined by the node-specific features $Z_i$ with weights $\theta$; neighbor influence $\beta \cdot m_i$ is computed from the current states of neighboring nodes; $m_i=\frac{1}{d_i}\sum_{j=1}^N A_{ij}X_j$ is the averaged neighbor influence with strength of peer effect $\beta$. Finally, the node specific conditional probability of the following form can be achieved through the modeling process:

$$\mathbb{P}(X_i = 1 \mid X_{-i})=\sigma(Z_i\theta + \beta \cdot m_i)$$

**Note:** there is a trade-off between individual features and neighbor influence, and the outcome depends on which effect is stronger. For example, If a node’s own features strongly favor 1, but neighbors are mostly 0, this means the node's own feature dominates. If a node has most of its neighbors being 1, the probability for itself being one increases.

#### Pseudo Code:

**STEP 1: Create:**
- Adjacency matrix $A \in \mathbb{R}^{N\times N}_{sym}$ (network frame)
- Covariate matrix $Z \in \mathbb{R}^{N\times p}$ (node features)
- Parameter vector $\theta \in \mathbb{R}^{p}$ (feature effect)
- Parameter scalar $\beta \in \mathbb{R}$ (peer effect)
- Number of iterations `num_iter`

**STEP 2: Initialize:**  
Initialize $X^{(0)}$ randomly through a distribution (e.g. Bernoulli(0.5))

**STEP 3: Gibbs Process:**  
For t = 1 to `num_iter`:  
For each node i:  
- Compute Neighbor Influence: $m_i = \sum_j A_{ij} \cdot X_j$  
- Compute Individual Effect: $k_i = f(Z_i)$ e.g. weighted combination of features  
- Compute Score: $S_i = k_i + \beta \cdot m_i$  
- Convert to Probability: $p_i = sigmoid(S_i)$  
- Sample New State: $X_i \sim Bernoulli(P_i)$  

#### Code:



---

## Informal Reference Section

Bayesian Data Analysis Third Edition, Gelman

High-Dimensional Bayesian Regularized Regression with the bayesreg Package (arXiv:1611.06649)
