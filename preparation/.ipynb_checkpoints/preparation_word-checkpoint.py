import itertools
import math
from mpmath import *
from multiprocessing import Pool
import numpy as np
import pickle
import time    

# -- load

with open("../data/sentence_embedding.pkl", "rb") as f:
    s_embd = pickle.load(f)
    
with open("../data/sentence_index.pkl", "rb") as f:
    s_index = pickle.load(f)
    
with open("../data/word_embedding.pkl", "rb") as f:
    w_embd = pickle.load(f)
    
with open("../data/word_index.pkl", "rb") as f:
    w_index = pickle.load(f)    
    
# -- calculate top 10 for each word

num_of_sent = s_embd.shape[0]
num_of_word = w_embd.shape[0]
q = num_of_word // 1000

top_10 = dict()
    
def get_top_10(w):
    print(w)
    A = w_embd - w_embd[w]
    B = A*A
    C = B.sum(1)
    L = np.argsort(C)[0:10]
    return(L)

def main(M):
    with Pool(32) as pool:
        X = pool.map(get_top_10,M)
    return X

s = "top_10_word_{}"

for j in range(0,q):
    result = main(range(1000*j,1000*(j+1)))
    Y = {1000*j+s : result[s] for s in range(0,len(result))}
    top_10 = top_10 | Y
    
result = main(range(q*1000,num_of_word))
Y = {result[s][0] : result[s] for s in range(0,len(result))}
top_10 = top_10 | Y
    
# -- calculate the embedding matrix by averaging word vectors constituting the sentence

#s_index_switch = {s_index[w]:w for w in s_index.keys()}
w_index_switch = {w_index[i]:i for i in range(0,num_of_word)}

s_w_index = dict()

for i in range(0,num_of_sent):
    w_indices = [w_index_switch[w] for w in s_index[i].split()]
    s_w_index[i] = [index for index in w_indices] 
    
def vec_gen(i):
    if len(s_w_index[i]) != 0:
        vec = np.mean(np.array([w_embd[j] for j in s_w_index[i]]),axis=0)
    else:
        vec = s_embd[i]
    return (i,vec)

def main():
    with Pool(32) as pool:
        X = pool.map(vec_gen,range(0,num_of_sent))
    return dict(X)

if __name__ == "__main__":
    mat = main()

sentence_embedding_mean = np.array([mat[i] for i in range(0,num_of_sent)])

with open("../data/sentence_embedding_mean.pkl", "wb") as f:
    pickle.dump(sentence_embedding_mean,f)
    
# -- calculate alpha

delta_NADP = {1.6: 100, 3.2: 30, 4.8: 15, 6.4: 8, 8.0: 7, 9.6: 6.4, 11.2: 5.5, 12.8: 5, 14.4: 4.8, 16.0: 4.5, 
              17.6: 4.2, 19.2: 4, 20.8: 3.7, 22.4: 3.5, 24.0: 3.4, 25.6: 3.3, 27.2: 3.0, 28.8: 3.0, 30.4: 3.0, 32.0: 3.0, 
              33.6: 3.0, 35.2: 3.0, 36.8: 3.0, 38.4: 3.0, 40.0: 3.0} 

# above paraeters should be given by users. Please choose the most suitable ones for your experiment.

n = num_of_word
v = "{}"
alpha = dict()
    
def Bf(e,u):
    e = mp.mpf(v.format(e))
    u = mp.mpf(v.format(u))
    value = ncdf(1/(2*u)-e*u)-exp(e)*ncdf(-1/(2*u)-e*u)
    return value
    
def B(e):
    k = delta_NADP[e]
    u = 1
    while Bf(e,u) >= 1/n**k:
        u += 1
    u = u-1
    i = 1
    while Bf(e,u+pow(10,-1)*i) >= 1/n**k:
        i += 1
    i = i-1
    u = u + pow(10,-1)*i
    i = 1
    while Bf(e,u+pow(10,-2)*i) >= 1/n**k:
        i += 1
    i = i-1
    u = u + pow(10,-2)*i
    i = 1
    while Bf(e,u+pow(10,-3)*i) >= 1/n**k:
        i += 1
    i = i-1
    u = u+pow(10,-3)*i
    i = 1
    while Bf(e,u+pow(10,-4)*i) >= 1/n**k:
        i += 1
    i = i-1
    u = u+pow(10,-4)*i
    i = 1
    while Bf(e,u+pow(10,-5)*i) >= 1/n**k:
        i += 1
    i = i-1
    u = u+pow(10,-5)*i
    i = 1
    while Bf(e,u+pow(10,-6)*i) >= 1/n**k:
        i += 1
    i = i-1
    u = u+pow(10,-6)*i
    i = 1
    while Bf(e,u+pow(10,-7)*i) >= 1/n**k:
        i += 1
    i = i-1
    u = u+pow(10,-7)*i
    i = 1
    while Bf(e,u+pow(10,-8)*i) >= 1/n**k:
        i += 1
    i = i-1
    u = u+pow(10,-8)*i
    i = 1
    while Bf(e,u+pow(10,-9)*i) >= 1/n**k:
        i += 1
    i = i-1
    u = u+pow(10,-9)*i
    i = 1
    while Bf(e,u+pow(10,-10)*i) >= 1/n**k:
        i += 1
    u = u+pow(10,-10)*i
    return u

for e in [8*i/5 for i in range(1,26)]:
    n = mp.mpf(v.format(n))
    print(e)
    alpha[e] = B(e)

with open("../data/alpha.pkl", "wb") as f:
    pickle.dump(alpha,f)

## -- construct the neighbourhood graph -- ##

# -- definition of functions

n = num_of_word
m = n // 10000
R = {i : range(10000*i,10000*(i+1)) for i in range(0,m)}
R[m] = range(10000*m,n)

def Jaccard_index(i,j):
    X_i, X_j = set(top_10[i]), set(top_10[j])
    its, uni = X_i & X_j, X_i | X_j
    jac = float(len(its)/len(uni))
    return jac

def neighbourhood(i,j):
    jac = Jaccard_index(i,j)
    # -- round down jac to the second decimal place
    jac_mod = math.floor(10*jac)/10
    if jac_mod >= 0.1:
        jac_mod = 0.1
    else:
        pass
    if (i in top_10[j][0:2] or j in top_10[i][0:2]) and (jac_mod >= 0.05):
        dist = np.linalg.norm(w_embd[i] - w_embd[j])
        print((i,j,dist))
        return (i,j,dist)
    else:
        pass

def collect(i,L):
    X = [(x[1],x[2]) for x in L if x[0] == i]
    X = sorted(X, key = lambda x: x[1])
    return (i,X)

def main(X,Y):
    with Pool(32) as pool:
        Z = pool.starmap(X,Y)
    return Z

# -- demonstrate the construction

L = []
    
for i, j in itertools.product(range(0,m+1),range(0,m+1)):
    domain = itertools.product(R[i],R[j])
    partial_L = main(neighbourhood,domain)
    partial_L = [x for x in partial_L if x != None]
    L += partial_L

domain = itertools.product(range(0,n),[L])
edge_data = main(collect,domain)
edges = dict(edge_data)

i = 1
r = set()
s = dict()
t = {0:set()}
nbd_graph = dict()

while True:
    s[i] = dict()
    m = min(set(range(0,n)) - r)
    z = {0:{m}}
    j = 1
    z[j] = set(z[0])
    for x in edges[m]:
        s[i][(m,x[0])] = x[1]
        z[j].add(x[0])
    r = r | z[j]
    j=j+1
    print(len(z[0]),len(z[1]),len(r))
    while len(z[j-1] - z[j-2]) > 0:
        z[j] = set(z[j-1])
        for m in list(z[j-1] - z[j-2]):
            for x in edges[m]:
                s[i][(m,x[0])] = x[1]
                z[j].add(x[0])
        r = r | z[j]
        j=j+1
    t[i] = set(r)
    if len(r) == n:
        break
    else:
        i=i+1
        
for j in range(1,i+1):
    nbd_graph[j] = (t[j] - t[j-1], max(s[j].values()))

# -- represent each sentence by index numbers    

covering = {sorted(list(nbd_graph[i][0]))[0]: sorted(list(nbd_graph[i][0])) for i in nbd_graph.keys()}

X = list()    
    
for i in covering.keys():
    if len(covering[i]) == 1:
        X.append(i)

s_w_index_mod = dict()

for i in range(0,num_of_sent):
    indices = [w_index_switch[w] for w in s_index[i].split()]
    s_w_index_mod[i] = [index for index in indices if index not in X] 
    
with open("../data/convert_num.pkl", "wb") as f:
    pickle.dump(s_w_index_mod,f)    
    
# -- determination of Delta for each vocabullary

Delta_list = list()
n = len(nbd_graph.keys())
    
for i in range(1,n+1):
    Delta_list.append(nbd_graph[i][1])
    
num_Delta = dict()

for i in range(1,n+1):
    for j in list(nbd_graph[i][0]):
        num_Delta[j] = nbd_graph[i][1]
    
with open("../data/num_Delta.pkl", "wb") as f:
    pickle.dump(num_Delta,f)








