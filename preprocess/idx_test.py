import numpy as np
import torch
text=torch.tensor([11,12,13,14,15,16,17,18,19])

idx=torch.tensor([8,7,6,5,4,3,2,1,0])
print("text:",text)
print("idx text:",text[idx])
print()

print("batch_text######################")
batch_text=text.unsqueeze(0).repeat(32,1)
print("batch_text:",np.shape(batch_text))

print("original batch text:",batch_text)
print("indexed batch text:",batch_text[:,idx])

print("batch_3_text######################")
batch_3_text=text.unsqueeze(0).unsqueeze(1).repeat(32, 3, 1)
print("batch_3_text:",np.shape(batch_3_text))
print("original batch text:",batch_3_text)
print("indexed batch text:",batch_3_text[:,:,idx])

print("batch_text&&idx######################")
print("original batch text:",batch_text.shape)
print("indexed batch text:",batch_text[:,idx].shape)

print("batch_3_text&&idx######################")
print("original batch text:",batch_3_text.shape)
print("indexed batch text:",batch_3_text[:,:,idx].shape)






