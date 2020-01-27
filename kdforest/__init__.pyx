#!python
#cython: language_level=3
#cython.wraparound=False
#cython.boundscheck=False
#cython.nonecheck=False
import linearity as ln
import random
import dikt


cdef class KDForest(object):
    cdef int n_trees
    cdef int has_graph
    cdef public object tree
    cdef public object graph
    cdef public object hc

    def __init__(self, filename=None, n_trees=1, graph=None):
        self.n_trees = n_trees
        self.has_graph = False

        if filename is not None:
            if ".dikt" in filename:
                self.tree = dikt.load(filename)
            else:
                import json
                self.tree = json.load(open(filename))

            if graph is not None:
                self.graph = dikt.load(graph)
                self.has_graph = True

    cpdef query(self, v, k=10):
        cdef dict count = {}
        cdef list neighbors
        cdef list added_neighbors
        cdef str char
        cdef int num_n
        cdef set avoid

        # convert query
        if isinstance(v, list):
            vec = ln.vector(v)
        elif not isinstance(v, ln.vector):
            vec = ln.vector(list(v))

        # case for 1 tree
        if self.n_trees == 1:
            neighbors, char = query(vec, self.tree)

            # complete query
            num_n = len(neighbors)
            avoid = {char}
            char = char[:-2]
            while num_n < k:
                added_neighbors, avoid = populate(
                    self.tree, char=char, avoid=avoid)
                neighbors += added_neighbors
                char = char[:-1]
                num_n = len(neighbors)
            return neighbors[:k]
        else:
            for i in range(self.n_trees):
                nn, char = query(vec, self.tree, char=f"{i}_0")
                for n in nn:
                    if n in count:
                        count[n] += 1
                    else:
                        count[n] = 1
            
            neighbors = sorted(count.items(), key=get_item, reverse=True)[:k]
            if self.has_graph:
                neighbors = self.get_neighbors_from_graph(neighbors[0][0], k)
        return neighbors

    cpdef get_neighbors_from_graph(self, object neighbor, int k):
        cdef neighbors = [neighbor] + self.graph[neighbor]
        j = 1
        if len(neighbors) < k:
            length = len(neighbors)
            while len(neighbors) < k and j < 2 * length:
                new_neighbors = self.graph[neighbors[j]]
                for idx in new_neighbors:
                    if idx not in neighbors:
                        neighbors.append(idx)
                        if len(neighbors) >= k:
                            break
                j += 1
        return neighbors[:k]

    cpdef build_index(self, list X, list ids, int limit=10, int samples=500):
        cdef dict endpoints = {}
        cdef dict tree = {}
        cdef int dim = X[0].size
        import time

        # start timer
        start_time = time.time()

        # convert X to a list of `linearity.vectors`
        if not isinstance(X[0], ln.vector):
            X = self.cast_to_vectors(X)

        # start subdivisions
        if self.n_trees == 1:
            subdivide(
                X, ids, dim, endpoints, tree,
                limit=limit, max_samples=samples)
            self.merge(endpoints, tree)
        else:
            from tqdm import tqdm
            many_endpoints = []
            many_trees = []
            for i in tqdm(range(self.n_trees), desc="building trees"):
                endpoints = {}
                tree = {}
                subdivide(
                    X, ids, dim, endpoints, tree,
                    limit=limit, max_samples=samples)
                many_endpoints.append(endpoints)
                many_trees.append(tree)
            self.merge_many(many_endpoints, many_trees)

        # stop timer
        elapsed_time = time.time() - start_time
        print(f"index built in {elapsed_time:.2f}s")

    cpdef merge(self, dict endpoints, dict tree):
        self.tree = {}
        for key, value in endpoints.items():
            self.tree[key + "e"] = value
        for key, value in tree.items():
            self.tree[key] = [int(value[0]), float(value[1])]

    cpdef merge_many(self, list e, list t):
        self.tree = {}
        for i in range(self.n_trees):
            for key, value in e[i].items():
                self.tree[f"{i}_{key}e"] = value
            for key, value in t[i].items():
                self.tree[f"{i}_{key}"] = value

    def save_index(self, filename, factor=250):
        assert hasattr(self, "tree")

        if ".dikt" in filename:
            N = len(self.tree)
            chunks = N // factor
            dikt.dump(self.tree, filename, chunks=chunks, compression=1)
        else:
            import json
            with open(filename, "w") as f:
                json.dump(self.tree, f)

    cpdef cast_to_vectors(self, X):
        for i in range(len(X)):
            X[i] = ln.vector(list(X[i]))
        return X


cpdef subdivide(
    list X, list ids, int dim,
    dict endpoints={}, dict tree={},
    str char="0", int limit=40,
    int max_samples=20000
):
    cdef list Y
    cdef list upper
    cdef list lower
    cdef list upper_ids
    cdef int size = len(X)

    if size <= limit:
        endpoints[char] = ids
    else:
        # compute axis with largest variance
        Y = random.sample(X, min(size, max_samples))
        idx, axis = ln.axis_of_max_variance(Y, dim)
        axis_median = axis.median()
        axis_median = ln.approximate(axis_median)

        # store information in tree
        tree[char] = [idx, axis_median]

        # sort vectors to the corresponding subspace
        upper, lower = [], []
        upper_ids, lower_ids = [], []
        for i in range(size):
            vector_ = X[i]
            vector_id = ids[i]
            if vector_.value[idx] >= axis_median:
                upper.append(vector_)
                upper_ids.append(vector_id)
            else:
                lower.append(vector_)
                lower_ids.append(vector_id)

        # recursively subdivide space
        subdivide(
            upper,
            upper_ids,
            dim,
            endpoints,
            tree,
            limit=limit,
            char=char + "1",
            max_samples=max_samples)
        subdivide(
            lower,
            lower_ids,
            dim,
            endpoints,
            tree,
            limit=limit,
            char=char + "0",
            max_samples=max_samples)


cpdef query(vec, object tree, str char="0"):
    cdef int idx
    cdef float threshold
    cdef list neighbors

    while True:
        try:
            idx, threshold = tree[char]
            if vec.value[idx] >= threshold:
                char += "1"
            else:
                char += "0"
        except KeyError:
            neighbors = tree[char + "e"]
            return neighbors, char + "e"


cpdef populate(tree, str char="0", avoid=set()):
    endpoint = char + "e"
    if endpoint in avoid:
        return [], avoid
    elif char not in tree:
        avoid.add(endpoint)
        return tree[endpoint], avoid

    n, a = populate(tree, char + "1", avoid=avoid)
    avoid = avoid.union(a)

    n2, a2 = populate(tree, char + "0", avoid=avoid)
    avoid = avoid.union(a2)
    return n + n2, avoid


def get_item(x):
    return x[1]
