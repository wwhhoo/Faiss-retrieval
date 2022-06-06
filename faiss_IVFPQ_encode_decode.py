# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss

m = 8
nlist = 2**m
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                  # 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
# Keys
list_nos = np.zeros(xb.shape[0])
list_nos = np.int64(list_nos)
# allocate memory
codes = np.empty((nb,index.code_size), dtype=np.uint8)
# encode
index.encode_multiple(nb, faiss.swig_ptr(list_nos), faiss.swig_ptr(xb), faiss.swig_ptr(codes), compute_keys =True)
print(codes)
# decode
xcodes = np.empty((nb,d), dtype=np.float32)
index.decode_multiple(nb, faiss.swig_ptr(list_nos), faiss.swig_ptr(codes), faiss.swig_ptr(xcodes))
print(xcodes)