#!/usr/bin/env python
# coding: utf-8

# # 快速排序

# In[1]:


def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]
    return arr
    
def quick_sort(arr, left=None, right=None):
    if not isinstance(left, int):
        left = 0
    if not isinstance(right, int):
        right = len(arr) - 1
    if left == right:
        return arr
    
    pivot = left
    i = left
    j = right
    while i < j:
        while i < j and arr[j] >= arr[pivot]:
            j -= 1
        if i < j:
            arr = swap(arr, pivot, j)
        while i < j and arr[i] <= pivot:
            i += 1
        if i < j:
            arr = swap(arr, i, pivot)
    if left <= pivot:
        quick_sort(arr, left, pivot)
    if pivot + 1 <= right:
        quick_sort(arr, pivot + 1, right)
    
    return arr


# In[2]:


arr = [2, 1, 4, 6, 5]

print(quick_sort(arr))

