from __future__ import(absolute_import,division,print_function,unicode_literals)

import io
import os

from surprise import BaselineOnly
from surprise import Dataset
from surprise import evaluate
from surprise import Reader
from surprise import KNNBaseline


def read_uid():
    uid={}
    mid={}
    file_name=os.path.expanduser('~')+"/Downloads/CSC522/toy/sample.data"
    with io.open(file_name,'r',encoding='ISO-8859-1') as f:
#        print "loading file"
        for line in f:
            line=line.split(',')
            if int(line[1]) in uid:
                uid[int(line[1])].update({int(line[0]):int(line[2])})
            else:
                uid[int(line[1])]={int(line[0]):int(line[2])}
#        print "done!"
    return uid


file_path=os.path.expanduser('~')+"/Downloads/CSC522/toy/sample.data"
reader=Reader(line_format='item user rating timestamp',sep=',')
data=Dataset.load_from_file(file_path,reader=reader)
data.split(n_folds=60)

trainset=data.build_full_trainset()
sim_options={'name':'pearson_baseline','user_based':True}
algo=KNNBaseline(sim_options=sim_options)
algo.train(trainset)

user_id='911'
user_inner_id=algo.trainset.to_inner_uid(user_id)
user_neighbors=algo.get_neighbors(user_inner_id,k=22)
user_neighbors=(algo.trainset.to_raw_uid(inner_id) for inner_id in user_neighbors)

print()
print('The 5 nearest neighbors of the userid %s are:'%user_id)
for userid in user_neighbors:
    print(userid)

perf=evaluate(algo,data,measures=['RMSE','MAE'])
print (perf)
