#!/usr/bin/env python
# coding: utf-8

# \begin{align}&max \sum_{i\in I}(\alpha\cdot D_{i,t} + \beta\cdot R_{i,t})\cdot x_{i,t}\\
# s.t.\ \ \ \ \ &\sum_{i\in I}\pi_{t}(D_{i,t})\cdot x_{i,t} \le B \\
# & \sum_{t\in T}x_{i,t} = 1\ and\ x_{i,t}\in \{0,1\}\end{align}
# 

# 标注：
# 
# - $D_{i,t}$: 第i个设备或第i种设备群在treatment $t$下的预估时长，等于base_model和uplift model的加和，
# 
#     $D_{i,t}=D_{i,t}^{Uplift} + D_{i,t_0}^{Base}$，$t_0\in T$，为不发补贴的策略
# - $R_{i,t}$: 同上，为第i个设备或第i种设备群在treatment $t$下的次日留存率
# - $x_{i,t}$: 第i个设备或第i种设备群是否采用treatment $t$，0为不采用，1为采用，所要求解的策略矩阵
# - $\pi_t$: $t$是treatment，为时间上限和金币上限的组合，$\pi$为在treatment $t$下，时长和活跃天数到接收金币的映射
# - $\alpha$和$\beta$: 权重参数，控制活跃天数和时长的比重

# In[ ]:




