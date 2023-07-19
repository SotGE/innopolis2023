#!/usr/bin/env python
# coding: utf-8

# In[2]:


print("Сорокин Максим Евгеньевич")


# In[25]:


class Text:
    def __init__(self, name):
        self.__name = name
    def hi(self, *args):
        print(f'Привет, {self.__name}')


# In[26]:


instance = Text('Сорокин Максим Евгеньевич')


# In[27]:


instance.hi()


# In[ ]:




