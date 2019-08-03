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
        vector[unordered_set[int]] query_multiset(int*, int) except +
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
    def query_matrix(self, np.ndarray[int, ndim=2, mode="c"] fp, np.ndarray[long, ndim=1, mode="c"] labels, int N, int total_size):
        multiset = self.c_lsh.query_multiset(&fp[0, 0], N)

        cdef total_count = 0
        cdef max_size = 0
        cdef int local_label = 0
        for idx in range(len(multiset)):
            local_label = labels[idx]
            multiset[idx].erase(local_label)
            total_count += len(multiset[idx])
            max_size = max(max_size, len(multiset[idx]))

        np_lsh = np.zeros([N, max_size], dtype=np.int64)
        np_lsh.fill(total_size)
        for bdx, item in enumerate(multiset):
            for ldx, index in enumerate(item):
                np_lsh[bdx, ldx] = index
        return np_lsh, total_count

    def clear(self):
        self.c_lsh.clear()
