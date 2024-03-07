#!/usr/bin/env python
# coding: utf-8

# # EDUNET FOUNDATION-Class Exercise Notebook

# ## LAB 2 - Blur faces present in a given image

# ### Load OpenCV library

# In[1]:


import cv2


# ### Load the cascade
# 

# In[2]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# ### Read the image

# In[3]:


img = cv2.imread('demo_face.jpg')


# ### Convert to grayscale
# 

# In[4]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ### Detect the faces
# 

# In[5]:


faces = face_cascade.detectMultiScale(gray, 1.1, 4)


# ### Draw the rectangle around each face
# 

# In[6]:


for (x, y, w, h) in faces:
    ROI = img[y:y+h, x:x+w]
    blur = cv2.GaussianBlur(ROI, (91,91),0) 
    img[y:y+h, x:x+w] = blur
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


# ### Show original image

# In[7]:


cv2.imshow('img', img)


# In[8]:


cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




