#!/usr/bin/env python
# coding: utf-8

# # 第K大的数

# In[1]:


def quick_sort(arr, l, r):
    pivot = l
    i, j = l, r
    print(arr[l:r+1])
    while i < j:
        while i < j and arr[pivot] >= arr[i]:
            i += 1
        while i < j and arr[pivot] <= arr[j]:
            j -= 1
        arr[i], arr[j] = arr[j], arr[i]
    arr[i], arr[pivot] = arr[pivot], arr[i]
    if l < i-1:
        quick_sort(arr, l, i-1)
    if r > i+1:
        quick_sort(arr, i+1, r)
    return arr


# In[2]:


arr = [2, 1, 4, 7, 6, 5]

quick_sort(arr, 0, len(arr)-1)

