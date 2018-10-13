import pickle
from pyltp import Parser
from pyltp import Postagger
import collections

import numpy as np


def adjacency_matrix(heads):
    s = len(heads)
    matrix = np.zeros([s, s])
    for i in range(s):
        if not heads[i] == 0:
            matrix[heads[i] - 1][i] = 1
    return matrix


def degree_matrix(heads):
    s = len(heads)
    matrix = np.zeros([s, s])
    coll = collections.Counter(heads)
    for i in range(s):
        d = 1
        if i + 1 in coll.keys():
            d += coll[i + 1]
        matrix[i][i] = d
    return matrix


def laplacian_matrix(heads):
    return degree_matrix(heads) - adjacency_matrix(heads)

def get_fiedler(sentence):
    postags = postagger.postag(sentence)
    arcs = parser.parse(sentence, postags)
    heads = [arc.head for arc in arcs]
    # print(heads)
    L = laplacian_matrix(heads)
    # print("Laplacian matrix:")
    # print(L)
    eigenvalue, eigenvectors = np.linalg.eig(L)
    # print("eigenvectors:")
    # print(eigenvectors)
    fiedler = eigenvectors[1]
    # print("Fiedler:")
    # print(fiedler)
    return fiedler
if __name__ == '__main__':
    mode="testa"
    fiedler_dict={}
    paser_path = "ltp/parser.model"
    tagger_path = "ltp/pos.model"

    parser = Parser()
    postagger = Postagger()

    parser.load(paser_path)
    postagger.load(tagger_path)

    data = pickle.load(open("../data/" + mode + "_seg.pkl", "rb"))
    size=len(data)
    for i in range(size):
        print("{} in {}".format(i,size))
        line=data[i]
        text=line[1]
        id=line[-1]
        print(text,id)
        if len(text) > 500:
            text=text[:500]
        fiedler=get_fiedler(text)
        fiedler_dict[id]=fiedler

    pickle.dump(fiedler_dict,open("../data/dep/"+mode+".fiedler.pkl","wb"))
    print("done !")
    parser.release()  # 释放模型
    postagger.release()


