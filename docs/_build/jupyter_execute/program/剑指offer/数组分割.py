#!/usr/bin/env python
# coding: utf-8

# # 数组分割

# ## 1. 数组分割后方差和最小
# 
# 
# ### 题目
# 
#     将一个随机数组分割成两部分，使得分割后的两部分的方差和最小。
# 
# 
# ### 解法
#     
#     方差表达式
# $$\begin{split}\sigma^2(x)&=E\left[\left(x-E\left[x\right]\right)^2\right]\\&=E\left[x^2-2xE\left[x\right]+E^2\left[x\right]\right]\\&=E\left[x^2\right]-E^2\left[x\right]\end{split}$$
# 
#     借用该表达式，可以递归地求得增加新的元素后的数组方差
# 
# 
# ### 代码

# In[1]:


def calVariance(nums):
    vars_list = []
    cum_val = 0
    cum_square = 0
    for i, num in enumerate(nums):
        cum_val += num
        cum_square += num**2
        var = cum_square/(i+1) - (cum_val/(i+1))**2
        vars_list.append(var)
    return vars_list
    
    
def minVariancePartial(nums):
    if not nums:
        return -1, -1
    
    left_vars = calVariance(nums[:-1])
    right_vars = calVariance(nums[::-1][:-1])[::-1]
    
    max_var = 0
    split_idx = 0
    for idx, (left_var, right_var) in enumerate(zip(left_vars, right_vars)):
        sum_var = left_var + right_var
        if sum_var > max_var:
            max_var = sum_var
            split_idx = idx+1
            
    return split_idx, max_var


# 测试
nums = [1, 2, 3, 4, 5]
minVariancePartial(nums)

