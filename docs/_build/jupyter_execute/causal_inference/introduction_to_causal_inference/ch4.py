#!/usr/bin/env python
# coding: utf-8

# # 4. Causal Models

# ## 4.1 The do-operator and Interventional Distributions
# 
# **potential outcome $Y(t)$的分布**
# 
# $$P(Y(t)=y)\triangleq P(Y=y|do(T=t)) \triangleq P(y|do(t))$$
# 
# **ATE(average treatment effect)**
# 
# $$\mathbb{E}\left[Y|do(T=1)\right]-\mathbb{E}\left[Y|do(T=0)\right]$$
# 
# 从普适性来说，我们会更关注所有的分布，例如$P(Y|do(T=t))$，而不是均值。如果能描述$P(Y|do(t))$那么自然就能描述$\mathbb{E}\left[Y|do(t)\right]$。
# 
# 从概念上讲，干预分布（如$P(Y|do(T=t))$）与客观分布（$P(Y)$)是很不同的。客观分布不需要do操作，即不需要实施任何实验。如果我们能把带有do的表达式$Q$转换为不带有do的表达式，那么$Q$就被称为identifiable。
# 
# - causal estimand: 含有do操作
# - statical estimand: 不含有do操作
# 
# $do(t)$出现在条件长条$|$之后，表示干预后的世界。例如，$\mathbb{E}\left[Y|do(t),Z=z\right]$表示的是子人群$Z=z$受到干预$t$后的期望。相反，$\mathbb{E}\left[Z=z\right]$只表示子人群受到正常treatment（干预前）的期望值。

# ## 4.2 The Main Assumption: Modularity
# 
# ```{admonition} Assumption 4.1 (Modularity / Independent Mechanisms / Invariance)
# :class: note
# If we intervene on a set of nodes $S\subseteq [n]$(denotes set $\{1,2,\cdots,n\}$), setting them to constants, then for all $i$, we have the following:
# 
# 1. If $i\notin S$, then $P(x_i|pa_i)$ remains unchanged.
# 2. If $i\in S$, then $P(x_i|pa_i)=1$ if $x_i$ is the value that $X_i$ was set to by the intervention; otherwise, $P(x_i|pa_i)=0$.
# ```
