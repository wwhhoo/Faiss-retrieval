# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from tempfile import mkdtemp, TemporaryDirectory
import os.path as path
import faiss
from time import time

d = 128
nlist = 256
m = 16
k = 4
with TemporaryDirectory() as temp_dir:
    filename = path.join(mkdtemp(), 'E:/James/Code/Temp/newfile.dat')
    fp = np.memmap(filename, dtype='float32', mode='w+', shape=(1000000000,d))
    start_time = time()
    for i in range (10):
        print(i+1)
        file = np.load(f"E:/James/Code/SIFT1B/SIFT100MRealNumber{i}.npy")
        fp[i*100000000:(i+1)*100000000] = file[:]
    del fp
    del file
    end_time = time()
    time_spend =  end_time-start_time
    print(f"load time : {time_spend//3600}小時,{(time_spend%3600)//60}分鐘,{(time_spend%3600%60)}秒")
    # check map data
    newfp = np.memmap(filename, dtype='float32', mode='r', shape=(1000000000,d))
    query = np.loadtxt("E:/James/Code/SIFT1B/Test.txt")
    query = np.float32(query)
    quantizer = faiss.IndexFlatL2(d) 
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    print("Index Training...")
    start_time = time()
    index.train(newfp)
    end_time = time()
    time_spend =  end_time-start_time
    print(f"Train time : {time_spend//3600}小時,{(time_spend%3600)//60}分鐘,{(time_spend%3600%60)}秒")
    PQindex_name = "index_1B_PQ.index"
    faiss.write_index(index, PQindex_name)