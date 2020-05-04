#!/usr/bin/env python
# coding: utf-8

# ## Pets Classification with Fast ai

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from fastai import *
from fastai.vision import *


# In[6]:


import os


# In[16]:


fname_imag = get_image_files(os.path.join('./images'))


# In[19]:




# In[20]:


path_img = os.path.join('./images/')
path_anno = os.path.join('./annotations')


# In[21]:


np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


# In[22]:


data = ImageDataBunch.from_name_re(path_img, fname_imag, pat, ds_tfms=get_transforms(), size=224, bs=16).normalize(imagenet_stats)


# In[23]:


#data.show_batch(rows=3, figsize=(7,6))


# In[24]:


#print(data.classes)


# In[26]:


#len(data.classes), data.c


# ### Training Resnet34

# In[27]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[28]:


learn.model


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:




