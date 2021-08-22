#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import List


# In[2]:


def findKthLargest(nums: List[int], k: int) -> int:
    def quick_sort(l, r):
        p = l
        i, j = l, r
        while i < j:
            while i < j and nums[p] >= nums[j]: j -= 1
            while i < j and nums[p] <= nums[i]: i += 1
            nums[i], nums[j] = nums[j], nums[i]
        nums[i], nums[p] = nums[p], nums[i]
        print(nums, i)
        if k == i+1: return nums[i+1]
        if k > i+1: return quick_sort(i+1, r)
        if k < i+1: return quick_sort(l, i-1)

    return quick_sort(0, len(nums)-1)


# In[3]:


nums = [3,2,3,1,2,4,5,5,6]
k = 4


# In[4]:


findKthLargest(nums, k)


# In[5]:


a = {'a': 1, 'b': 2}


# In[6]:


list(a)


# In[ ]:




