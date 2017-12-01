import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Merge, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adamax
from keras.callbacks import EarlyStopping, ModelCheckpoint

f = open('config', 'r')
x = []
for line in f:
	x.append(line.split()[1])
train, test, batch_size, epoch = 'data/' + x[0], 'data/' + x[1], int(x[2]), int(x[3])

def loadNetflixData(file):
	user, movie, rate = [], [], []
	f = open(file, "r")  
	for line in f:  
	    x = line.split(',')
	    movie.append(int(x[0]))
	    user.append(int(x[1]))
	    rate.append(int(x[2]))
	f.close() 
	return np.array(movie), np.array(user), np.array(rate)

def saveArray(arr, file):
	f = open(file, "w")
	for i in range(len(arr)):
		for j in range(len(arr[0])):
			f.write(str(arr[i][j]))
			f.write(' ')
		f.write('\n')
	f.close()

movie_train, user_train, rate_train = loadNetflixData(train) 
movie_test, user_test, rate_test = loadNetflixData(test) 

# movie_count = max(movie_test) + 2
# user_count = max(user_test) + 2
movie_count = 500
user_count = 200000
model_left = Sequential()
model_left.add(Embedding(movie_count, 20, input_length=1))
model_right = Sequential()
model_right.add(Embedding(user_count, 20, input_length=1))
model = Sequential()
model.add(Merge([model_left, model_right], mode='concat'))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.7))
model.add(Activation('sigmoid'))
model.add(Dense(64))
model.add(Dropout(0.7))
model.add(Activation('sigmoid'))
model.add(Dense(64))
model.add(Dropout(0.7))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')



model.fit([movie_train, user_train], rate_train, batch_size=batch_size, epochs=epoch, validation_split=.1, verbose=2)
# model.save('model')
print(model_left.layers[0].get_weights())
saveArray(model_left.layers[0].get_weights()[0], 'movie_weights')
saveArray(model_right.layers[0].get_weights()[0], 'user_weights')
# model = load_model('model')
score = model.evaluate([movie_test, user_test], rate_test, batch_size=int(batch_size / 5))
print(math.sqrt(score))