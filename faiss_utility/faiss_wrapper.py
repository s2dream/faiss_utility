import faiss
import numpy as np
import os
from enum import Enum, auto


class INDEX_TYPE(Enum):
    FLAT_L2 = auto()


class faiss_wrapper:
    def check_np_array_type(self, target):
        return (type(target).__module__ == np.__name__)

    def get_size(self):
        return int(self.index.ntotal)


class faiss_index_wrapper(faiss_wrapper):
    def __init__(self, dim_size, index_type=INDEX_TYPE.FLAT_L2):
        self.index = self.generate_a_new_indexer(dim_size=dim_size, index_type=index_type)

    def generate_a_new_indexer(self, dim_size, index_type):
        if index_type == INDEX_TYPE.FLAT_L2:
            return faiss.IndexFlatL2(dim_size)
        else:
            return faiss.IndexFlatL2(dim_size)

    def add(self, data):
        if not self.check_np_array_type(data):
            print("the input data type is not numpy array")
            return
        data = data.astype('float32')
        self.index.add(data)
        print("new data is successfully added, index size:{0}".format(self.get_size()))

    def write_index(self, path):
        faiss.write_index(self.index, path)

class faiss_searcher_wrapper(faiss_wrapper):
    def __init__(self, index_path):
        if not os.path.exists(index_path):
            raise FileNotFoundError
        self.index = faiss.read_index(index_path)

    def search(self, query, num_results=10):
        if not self.check_np_array_type(query):
            print("the query type is not numpy array")
            return
        if len(query.shape) > 2:
            print(query)
            print("the query has more than two dimensions")
            return
        print(query.shape)
        query = query.astype('float32')
        distances, indices = self.index.search(query, num_results)
        print(distances)
        print(indices)


# test main
if __name__ == "__main__":
    INDEX_PATH = "./test_index.index"
    DIM_SIZE = 512
    DATA_SIZE = 600

    ## construction and saving index
    faiss_index_instance = faiss_index_wrapper(DIM_SIZE)
    data = np.random.rand(DATA_SIZE, DIM_SIZE)
    faiss_index_instance.add(data)
    faiss_index_instance.write_index(INDEX_PATH)

    faiss_searcher_inst = faiss_searcher_wrapper(INDEX_PATH)
    faiss_searcher_inst.search(data[:1])  # query shape: [num_of_query, dim_size]


