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
import json

m = 8
k = 4
bits = 8
nlist = 2**bits # for each m, how many candidate you want to search

# Set OPQ rotation matrix
opq_index = faiss.ProductQuantizer(d, m, bits)
opq = faiss.OPQMatrix(d, m)
opq.pq = opq_index
opq.train(xb)
# save and load Matrix
faiss.write_VectorTransform(opq, "opq.opq")
opq = faiss.read_VectorTransform("opq.opq")
# new query and data
xq = opq.apply_py(xq)
xb = opq.apply_py(xb)

# Use rotated data to train PQ
quantizer = faiss.IndexFlatL2(d)  
PQindex = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
PQindex.train(xb)
PQindex_name = "index_1M_OPQ.index"
faiss.write_index(PQindex, PQindex_name)
# opq.reverse_transform
