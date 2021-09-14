#!/usr/bin/env python
# coding: utf-8

# # 常用激活函数

# In[1]:


import numpy as np
import altair as alt
import pandas as pd


# ## 1. Sigmoid
# 
# ### 1.1 表达式
# 
# $$
# f(x)=\frac{1}{1+e^{-x}}
# $$ (sigmoid_function)
# 
# $f(x)$的取值范围为$(0, 1)$
# 
# ### 1.2 导数式
# 
# $$
# \begin{split}
# f^\prime(x)&=\frac{e^{-x}}{(1+e^{-x})^2} \\
# &=\frac{1}{1+e^{-x}}\cdot \frac{e^{-x}}{1+e^{-x}} \\
# &=f(x)\cdot \left(1-f(x)\right)
# \end{split}
# $$ (sigmoid_derivative)
# 
# $f^\prime(x)$的最大值为$f^\prime(0)=0.25$
# 

# In[2]:


# toy_data
x = np.linspace(-5, 5)
y = 1.0/(1.0+np.exp(-x))
dy = y * (1 - y)

df_y = pd.DataFrame.from_dict({
    "x": x, 
    "y": y, 
})
df_y["tag"] = "y"

df_dy = pd.DataFrame.from_dict({
    "x": x,
    "y": dy
})

df_dy["tag"] = "y'"

data = pd.concat([df_y, df_dy])

# plot
alt.Chart(data, title="Sigmoid Function").mark_line().encode(
    x="x:Q",
    y="y:Q",
    color="tag:N"
).properties(
    width=500,
    height=100
)


# ### 1.3 问题
# 
# 1. $Sigmoid$的导数小于等于0.25，反向传播的过程中容易出现梯度消失

# ## 2. ReLU
# 
# ### 2.1 表达式
# \begin{align}
# f(x)=\begin{cases} 
# x,  & \mbox{if }x\gt 0 \\
# 0, & \mbox{if }x\le 0
# \end{cases}
# \end{align} (relu_function)
# 
# ### 2.2 导数式
# \begin{align}
# f^\prime(x)=\begin{cases} 
# 1,  & \mbox{if }x\gt 0 \\
# 0, & \mbox{if }x\le 0
# \end{cases}
# \end{align} (relu_derivative1)

# In[3]:


# toy_data
x = np.linspace(-5, 5)
y = np.copy(x)
y[x<=0] = 0
dy = np.copy(x)
dy[x<=0] = 0
dy[x>0] = 1

df_y = pd.DataFrame.from_dict({
    "x": x, 
    "y": y, 
})
df_y["tag"] = "y"

df_dy = pd.DataFrame.from_dict({
    "x": x,
    "y": dy
})

df_dy["tag"] = "y'"

data = pd.concat([df_y, df_dy])

# plot
alt.Chart(data, title="ReLU Function").mark_line().encode(
    x="x:Q",
    y="y:Q",
    color="tag:N"
).properties(
    width=300,
    height=300
)


# ### 2.3 问题
# 
# 1. 输入$x$小于等于0时，导数为0时，后续的参数不会被更新
# 
# 
# ### 2.4 可能解决方法
# 
# 1. $softplus$函数（实验效果与ReLU差不多）
# 
# - 表达式
# 
# $$
# f(x)=log(1+e^x)
# $$ (softplus_function)
# 
# - 导数式
# 
# $$
# f^\prime(x)=\frac{e^x}{1+e^x}
# $$ (softplus_derivative)
# 

# In[4]:


# toy_data
x = np.linspace(-5, 5)
y = np.log(1+np.exp(x))
dy = np.exp(x) / (1+np.exp(x))

df_y = pd.DataFrame.from_dict({
    "x": x, 
    "y": y, 
})
df_y["tag"] = "y"

df_dy = pd.DataFrame.from_dict({
    "x": x,
    "y": dy
})

df_dy["tag"] = "y'"

data = pd.concat([df_y, df_dy])

# plot
alt.Chart(data, title="SoftPlus Function").mark_line().encode(
    x="x:Q",
    y="y:Q",
    color="tag:N"
).properties(
    width=300,
    height=300
)


# 2. 将bias设大，可以让大部分nerual的值大于0（实验结果并不好）
# 
# 
# **结论：ReLU效果已经非常好了**
# 
# 
# ### 2.5 ReLU的变种
# 
# 1. Leaky ReLU
# 
# \begin{align}
# f(x)=\begin{cases} 
# x,  & \mbox{if }x\gt 0 \\
# 0.01x, & \mbox{if }x\le 0
# \end{cases}
# \end{align} (leaky_relu_function)
# 
# 为什么是0.01?
# 
# 2. Parametric ReLU
# 
# \begin{align}
# f(x)=\begin{cases} 
# x,  & \mbox{if }x\gt 0 \\
# \alpha\cdot x, & \mbox{if }x\le 0
# \end{cases}
# \end{align} (parametric_relu_function)
# 
# $\alpha$当作参数被训练

# ## 3. Maxout
# 
# ### 3.1 表达式
# 
# $$
# z=max\left\{z_1, z_2, ..., z_k\right\}
# $$ (maxout_function)
# 
# ```
# ReLU是一种特殊情况
# ```

# In[5]:


# toy_data
x = np.linspace(-5, 5)
z1 = x
z2 = np.zeros_like(x) -0.2
z = np.maximum(z1, z2) - 0.2

df_z1 = pd.DataFrame.from_dict({
    "x": x, 
    "y": z1, 
})
df_z1["tag"] = "z1"

df_z2 = pd.DataFrame.from_dict({
    "x": x,
    "y": z2
})

df_z2["tag"] = "z2"

df_z = pd.DataFrame.from_dict({
    "x": x,
    "y": z
})

df_z["tag"] = "z"


data = pd.concat([df_z1, df_z2, df_z])

# plot
alt.Chart(data, title="Maxout Function").mark_line().encode(
    x="x:Q",
    y="y:Q",
    color="tag:N"
).properties(
    width=300,
    height=300
)


# ## 4. 损失函数Softmax

# ### 4.1 表达式
# 
# $$
# f(x_i)=\frac{e^{x_i}}{\sum_j{e^{x_j}}}
# $$ (softmax_function)
# 
# **特性：**
# 
# - $1 > y_i > 0$
# 
# - $\sum_iy_i=1$
# 
# - 每个值之间的大小排序不变
# 
# - 每个值与其他值均有关
# 
# 
# ### 4.2 导数式
# 
# - $x_i$自身微分
# \begin{align}
# \begin{split}\frac{\partial{f(x_i)}}{\partial{x_i}}&=\frac{e^{x_i}\cdot \sum_j{e^{x_j}} - e^{x_i}\cdot e^{x_i}}{\left(\sum_j{e^{x_j}}\right)^2} \\
# &=f(x_i)\cdot\left(1-f(x_i)\right)
# \end{split}
# \end{align} (softmax_derivative1)
# 
# - 当$i\ne k$时
# \begin{align}
# \begin{split}\frac{\partial{f(x_i)}}{\partial{x_k}}&=-e^{x_i}\cdot\frac{e^{x_k}}{\left(\sum_j{e^{x_j}}\right)^2} \\
# &=-f(x_i)\cdot f(x_k)
# \end{split}
# \end{align} (softmax_derivative1)
