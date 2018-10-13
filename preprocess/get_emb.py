import pickle

import numpy as np


def txt2pkl():
    w2v={}
    sg=open("../data/emb/sgns.sogou.word",encoding="utf-8").readlines()
    for line in sg:
        sp=line.replace("\n","").split(" ")
        w=sp[0]
        v=[float(c) for c in sp[1:-1]]
        w2v[w] = v
        print(w,v)
    pickle.dump(w2v,open("../data/emb/w2v.pkl","wb"))
    print("done !")
def id2vec():
    w2v=pickle.load(open("../data/emb/w2v.pkl","rb"))
    word2id=pickle.load(open("../data/word2id.obj","rb"))
    empty=[]
    id2emb={}
    for k in word2id.keys():
        try:
            id2emb[word2id[k]] = w2v[k]
            print(w2v[k])
        except:
            id2emb[word2id[k]] = list(np.random.uniform(-0.1,0.1,300)) # 将不包含的词语初始化在-0.1~0.1之间
            empty.append(k)
            print(k)
    pickle.dump(id2emb,open("../data/emb/id2w.pkl"))
    print("done !")

def id2vec():
    w2v=pickle.load(open("../data/emb/w2v.pkl","rb"))
    word2id=pickle.load(open("../data/word2id.obj","rb"))
    empty=[]
    id2emb={}
    for k in word2id.keys():
        try:
            id2emb[word2id[k]] = w2v[k]
            print(w2v[k])
        except:
            id2emb[word2id[k]] = list(np.random.uniform(-0.1,0.1,300)) # 将不包含的词语初始化在-0.1~0.1之间
            empty.append(k)
            print(k)
        if word2id[k] == 0:
            id2emb[0] = list(np.zeros(300))
    pickle.dump(id2emb,open("../data/emb/id2v.pkl","wb"))
    print("done !")


# txt2pkl()
# id2vec()
def get_emb_mat(id2v_path="../data/emb/id2v.pkl"):
    id2v=pickle.load(open(id2v_path,"rb"))
    id_list=sorted(list(id2v.keys()))
    embedding_matrix=[id2v[id] for id in id_list]
    return np.array(embedding_matrix)

if __name__ == '__main__':
    # id2vec()
    emb=get_emb_mat() # emb[0] == emb[<PAD>]
    for l in emb[:2]:
        print(l)
    word2id = pickle.load(open("../data/word2id.obj", "rb"))
    for k in word2id.keys():
        if word2id[k] == 0:
            print("id0:",k)
    id2v = pickle.load(open("../data/emb/id2v.pkl", "rb"))
    print("id2v0:",id2v[0])