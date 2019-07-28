from libcpp.unordered_set cimport unordered_set
from libcpp cimport bool
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
import cython

cdef extern from "LSH.h":
    cdef cppclass LSH:
        LSH(int, int, int) except +
        void remove(int*, int) except +
        void insert(int*, int) except +
        void insert_multi(int*, int) except +
        unordered_set[int] query(int*) except +
        unordered_set[int] query_multi(int*, int) except +
        void query_multi_mask(int*, float*, int, int) except +
        void clear()

cdef class pyLSH:
    cdef LSH* c_lsh

    def __cinit__(self, int K, int L, int THREADS):
        self.c_lsh = new LSH(K, L, THREADS)

    def __dealloc__(self):
        del self.c_lsh

    @cython.boundscheck(False)
    def remove(self, np.ndarray[int, ndim=1, mode="c"] fp, int item_id):
        self.c_lsh.remove(&fp[0], item_id)

    @cython.boundscheck(False)
    def insert(self, np.ndarray[int, ndim=1, mode="c"] fp, int item_id):
        self.c_lsh.insert(&fp[0], item_id)

    @cython.boundscheck(False)
    def insert_multi(self, np.ndarray[int, ndim=2, mode="c"] fp, int N):
        self.c_lsh.insert_multi(&fp[0, 0], N)

    @cython.boundscheck(False)
    def query(self, np.ndarray[int, ndim=1, mode="c"] fp):
        return self.c_lsh.query(&fp[0])

    @cython.boundscheck(False)
    def query_multi(self, np.ndarray[int, ndim=2, mode="c"] fp, int N):
        return self.c_lsh.query_multi(&fp[0, 0], N)

    @cython.boundscheck(False)
    def query_multi_mask(self, np.ndarray[int, ndim=2, mode="c"] fp, np.ndarray[float, ndim=2, mode="c"] mask, int M, int N):
        return self.c_lsh.query_multi_mask(&fp[0, 0], &mask[0,0], M, N)

    @cython.boundscheck(False)
    def accidental_match(self, np.ndarray[long, ndim=1, mode="c"] labels, set samples, int N):
        for idx in range(N): 
            if labels[idx] in samples:
                samples.remove(labels[idx])

    @cython.boundscheck(False)
    def multi_label(self, np.ndarray[long, ndim=2, mode="c"] labels, set samples):
        M = labels.shape[0]
        K = labels.shape[1]
        label2idx = dict()
        label_list = list()
        batch_prob = list()

        # remove accidental hits from samples
        # create label list
        # create label to index dictionary
        for idx in range(M): 
            count = 0
            for jdx in range(K): 
                l = labels[idx, jdx]
                if l == -1:
                    break
                elif l in samples:
                    samples.remove(l)
                count += 1
                if l not in label2idx:
                    label2idx[l] = len(label_list)
                    label_list.append(l)
            batch_prob.append(1.0 / count)

        sample_list = label_list + list(samples)

        # create probability distribution
        result = np.zeros([M, len(sample_list)], dtype=np.float32)
        for idx in range(M): 
            for jdx in range(K): 
                l = labels[idx, jdx]
                if l == -1:
                    break
                else:
                    result[idx, label2idx[l]] = batch_prob[idx]
        return sample_list, result

    def clear(self):
        self.c_lsh.clear()
