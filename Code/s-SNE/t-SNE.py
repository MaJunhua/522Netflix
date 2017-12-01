import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def t_SNE(X):
	X_embedded = TSNE(n_components=2).fit_transform(X)
	return X_embedded


f = open("user_weights", "r")  
data = []
for line in f:  
    a = line.split()
    b = list(map(lambda x: float(x), a))
    data.append(b)
f.close() 
# print data
data_2d = t_SNE(data)
x, y = [], []
for i in range(len(data_2d)):
	x.append(data_2d[i][0])
	y.append(data_2d[i][1])

plt.plot(x, y, '.')
plt.show()

