import numpy as np
import tensorflow as tf
# from sklearn.neighbors import NearestNeighbors
from math import exp
import math
from random import shuffle, sample, random
MACHINE_EPSILON = np.finfo(np.double).eps
#get data

movie_id = []
f = open("data/movies.txt", "r")  
for line in f:  
    a = line.split()
    b = list(map(lambda x: float(x), a))
    movie_id.append(b)
movie_id = np.array(movie_id)
f.close() 


users_id = []
f = open("data/users.txt", "r")  
for line in f: 
    entityid.append(line)
f.close() 

rate = []
f = open("data/rates.txt", "r")  
for line in f:  
    a = line.split()
    b = list(map(lambda x: int(x), a))
    rate.append(b)
rate = np.array(rate)


# for i in range(len(classi)):
#     if(int(classi[i]) not in class_dic):
#         class_dic[int(classi[i])] = []
#     class_dic[int(classi[i])].append(i)


data_num = len(entityid)
dimention = 2
neg_sample = 1
print(data_num, dimention)
dimension = 50

low_embedded = tf.Variable(tf.random_uniform([data_num, dimension], -math.sqrt(6)/2, math.sqrt(6)/2))
low_relation = tf.Variable(tf.random_uniform([len(relation_vec), dimension], -math.sqrt(6)/2, math.sqrt(6)/2))


triple_batch = tf.placeholder(tf.int32, [None, 3])
neg_triple_batch = tf.placeholder(tf.int32, [None, 3])
#pos

movie = tf.gather(low_embedded, triple_batch[:, 0])
user = tf.gather(low_relation, triple_batch[:, 1])
rate = tf.gather(low_embedded, triple_batch[:, 2])
triple_distance = tf.reduce_sum(tf.abs(h_entity + relation - t_entity), 1)

neg_h_entity = tf.gather(low_embedded, neg_triple_batch[:, 0])
neg_relation = tf.gather(low_relation, neg_triple_batch[:, 1])
neg_t_entity = tf.gather(low_embedded, neg_triple_batch[:, 2])
neg_triple_distance = tf.reduce_sum(tf.abs(neg_h_entity + neg_relation - neg_t_entity), 1)


# print(data_num)
# positive = tf.mul(p, pair_weight)
# objective = -(tf.reduce_sum(pair_weight * tf.log(p)))
# objective = -(1*tf.reduce_sum(tf.log(triple_p) * tri_weight + tf.log(1-neg_triple_p) * tri_weight))
objective = tf.reduce_sum(tf.maximum(tf.sqrt(triple_distance)  - tf.sqrt(neg_triple_distance) + 1, 0))

# objective = -(tf.reduce_sum(pair_weight*(tf.log(p)+1*tf.log(1-neg_p))) + 1*tf.reduce_sum(tf.log(triple_p) * tri_weight + tf.log(1-neg_triple_p) * tri_weight))
# objective = -(tf.reduce_sum(pair_weight*(tf.log(p)+5*tf.log(1-neg_p))+ 0.003*tf.reduce_sum(tf.log(1-neg_triple_p)) + 0.003*tf.reduce_sum(tf.log(triple_p)))
# objective = -(tf.reduce_sum(pair_weight * tf.log(p)))
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(objective)
init = tf.initialize_all_variables()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    index = list(range(len(triple)))
    batch = 100
    batch_size = int(len(triple) / batch)

    # Training cycle
    for epoch in range(1000):
        shuffle(index)
        neg_triples = neg_triple(triple)
        # Loop over all batches
        loss = 0
        for i in range(batch):
            
            # print(len(triple), len(triple_weight))
            batch_triple = triple[index[i*batch_size:(i+1)*batch_size]]
            batch_neg_triple = neg_triples[index[i*batch_size:(i+1)*batch_size]]
            # print(batch_triple[:10], batch_pair[:10])
            opt, le, lr, obj= sess.run([optimizer, low_embedded, low_relation, objective], feed_dict={triple_batch:batch_triple, neg_triple_batch:batch_neg_triple})
            loss += obj
        # print(lr)
        print ("epoch", epoch, "loss", loss)
        file_object = open('movie_embedding.txt', 'w')
        for i in le:
            for j in i:
                # print j
                file_object.write(str(j))
                file_object.write(' ')
            file_object.write('\n')
        file_object.close()
        file_object = open('user_embedding.txt', 'w')
        for i in lr:
            for j in i:
                # print j
                file_object.write(str(j))
                file_object.write(' ')
            file_object.write('\n')
        file_object.close()

    print("Optimization Finished!")
