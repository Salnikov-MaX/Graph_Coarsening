import numpy as np
import pygsp as gsp
import scipy as sp
import graph_coarsening.graph_utils as graph_utils


def coarsen(G, K=10, r=0.5, max_error=None):
    r = np.clip(r, 0, 0.999)
    N = G.N

    # current and target graph sizes
    n, n_target = N, np.ceil((1 - r) * N)

    # how much more we need to reduce the current graph
    r_cur = np.clip(1 - n_target / n, 0.0, 0.99)
    weights = get_proximity_measure(G)
    coarsening_list, error_list = matching_greedy(G, weights=weights, r=r_cur, k=K, max_error=max_error)

    return coarsening_list, error_list


def get_coarsed_graph(G, coarsening_list):
    Gall = []
    Gall.append(G)

    iC = get_coarsening_matrix(G, coarsening_list)
    C = sp.sparse.eye(G.N, format="csc")
    C = iC.dot(C)
    
    
    Wc = graph_utils.zero_diag(coarsen_matrix(G.W, iC))  # coarsen and remove self-loops
    Wc = (Wc + Wc.T) / 2  # this is only needed to avoid pygsp complaining for tiny errors

    if not hasattr(G, "coords"):
        Gc = gsp.graphs.Graph(Wc)
    else:
        Gc = gsp.graphs.Graph(Wc, coords=coarsen_vector(G.coords, iC))
   
    Gall.append(Gc)

    return Gall


def matching_greedy(G, weights, r=0.4, k=5, Uk=None, lk=None, max_error=None):

    N = G.N

    if (Uk is not None) and (lk is not None) and (len(lk) >= k):
        U, l = Uk, lk
    elif hasattr(G, "U"):
        U, l = G.U, G.e
    else:
        l, U = sp.sparse.linalg.eigsh(G.L, k=k, which="SM", tol=1e-3)

    C_ = sp.sparse.eye(N, format="csc")

    # the edge set
    edges = np.array(G.get_edge_list()[0:2])
    M = edges.shape[1]

    #idx = np.argsort(-weights) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    idx = np.argsort(weights)

    # idx = np.argsort(weights)[::-1]
    edges = edges[:, idx]

    # the candidate edge set
    candidate_edges = edges.T.tolist()

    # the matching edge set (this is a list of arrays)
    matching = []

    # which vertices have been selected
    marked = np.zeros(N, dtype=np.bool_)

    n, n_target = N, (1 - r) * N

    bad_iter_num = 0

    error_list = []

    while len(candidate_edges) > 0:
        if len(candidate_edges) == 0:
            print('All available graph edges are coarsened')
            break

        # pop a candidate edge
        [i, j] = candidate_edges.pop(0)

        # check if marked
        if any(marked[[i, j]]):
            continue

        marked[[i, j]] = True
        n -= 1

        # -------------------------------------------------------------------
    
        coarsening_list = matching.copy()
        coarsening_list.append(np.array([i, j]))
        iC = get_coarsening_matrix(G, np.array(coarsening_list))
        C = iC.dot(C_)
        
        metrics = coarsening_quality(N, l, G.L, C, kmax=k)
        error_eigenvalue = np.nanmean(metrics['error_eigenvalue'])
        error_list.append(error_eigenvalue)
        print(error_eigenvalue)


        if max_error:
            if error_eigenvalue > max_error:
                bad_iter_num+=1
                continue
            elif bad_iter_num > 10:
                print('Coarsining treshold was reached')
                break
            
            bad_iter_num = 0

        # -------------------------------------------------------------------

        # add it to the matching
        matching.append(np.array([i, j]))

        # termination condition
        if n <= n_target:
            break

    return np.array(matching), error_list


def coarsen_vector(x, C):
    return (C.power(2)).dot(x)


def coarsen_matrix(W, C):
    # Pinv = C.T; #Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return (Pinv.T).dot(W.dot(Pinv))


def get_coarsening_matrix(G, partitioning):
    """
    This function should be called in order to build the coarsening matrix C.

    Parameters
    ----------
    G : the graph to be coarsened
    partitioning : a list of subgraphs to be contracted

    Returns
    -------
    C : the new coarsening matrix

    Example
    -------
    C = contract(gsp.graphs.sensor(20),[0,1]) ??
    """

    # C = np.eye(G.N)
    C = sp.sparse.eye(G.N, format="lil")

    rows_to_delete = []
    for subgraph in partitioning:

        nc = len(subgraph)

        # add v_j's to v_i's row
        C[subgraph[0], subgraph] = 1 / np.sqrt(nc)  # np.ones((1,nc))/np.sqrt(nc)

        rows_to_delete.extend(subgraph[1:])

    # delete vertices
    # C = np.delete(C,rows_to_delete,0)

    C.rows = np.delete(C.rows, rows_to_delete)
    C.data = np.delete(C.data, rows_to_delete)
    C._shape = (G.N - len(rows_to_delete), G.N)

    C = sp.sparse.csc_matrix(C)

    # check that this is a projection matrix
    # assert sp.sparse.linalg.norm( ((C.T).dot(C))**2 - ((C.T).dot(C)) , ord='fro') < 1e-5

    return C


def coarsening_quality(N, l, L, C, kmax=30):
    # I = np.eye(N)

    l[0] = 1
    linv = l ** (-0.5)
    linv[0] = 0
    # l[0] = 0 # avoids divide by 0

    # below here, everything is C specific
    n = C.shape[0]
    #Pi = C.T @ C
    #S = graph_utils.get_S(G).T
    Lc = C.dot((L).dot(C.T))
    #Lp = Pi @ G.L @ Pi

    if kmax > n / 2:
        [Uc, lc] = graph_utils.eig(Lc.toarray())
    else:
        lc, Uc = sp.sparse.linalg.eigsh(Lc, k=kmax, which="SM", tol=1e-3)

    if not sp.sparse.issparse(Lc):
        print("warning: Lc should be sparse.")

    #metrics = {"r": 1 - n / N, "m": int((Lc.nnz - n) / 2)}
    metrics = {}

    kmax = np.clip(kmax, 1, n)

    # eigenvalue relative error
    metrics["error_eigenvalue"] = np.abs(l[:kmax] - lc[:kmax]) / l[:kmax]
    metrics["error_eigenvalue"][0] = 0

    return metrics


def get_proximity_measure(G):
    edges = np.array(G.get_edge_list()[0:2])
    weights = np.array(G.get_edge_list()[2])
    M = edges.shape[1]
    proximity = np.zeros(M, dtype=np.float32)

    # heuristic for mutligrid
    wmax = np.array(np.max(G.W, 0).todense())[0] + 1e-5
    for e in range(0, M):
        proximity[e] = weights[e] / max(wmax[edges[:, e]])  # select edges with large proximity
    return proximity



