import numpy as np

a = np.random.randint(0,2,(3,4))
b = np.random.randint(0,2,(4,3))
print(a)
print(b)

print(a@b)
print(np.dot(a,b))