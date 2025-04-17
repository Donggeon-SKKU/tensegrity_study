import sympy as sp
import numpy as np
from scipy.optimize import linprog
import networkx as nx

# Sigma 생성하기
def get_Sigma(i,num_member):
    num_component = num_member
    num_lambda = i
    num_gamma = num_component - num_lambda

    # lambda_vars = {}
    component = []
    count_lambda = 0
    count_gamma = 0

    for n in range(1,num_lambda+1):
        # lambda_name = f'l{n}'
        lambda_symbol = sp.symbols(f'l{n}')
        component.append(lambda_symbol)
        count_lambda += 1

    for m in range(1,num_gamma+1):
        # gamma_name = f'g{m}'
        gamma_symbol = sp.symbols(f'g{m}')
        component.append(gamma_symbol)
        count_gamma +=1


    Sigma = np.diag(component)
    return Sigma, count_lambda, count_gamma



def LP_prog_feasible(M,C,Sigma,count_lambda,count_gamma):
    A_eq = np.kron(C.T,M)
    # print(f"A_eq shape: {A_eq.shape}")

    vec_Sigma = Sigma.reshape(-1,order='F')

    zero_indices = [i for i, value in enumerate(vec_Sigma) if value == 0]
    A_new = np.delete(A_eq, zero_indices, axis=1)

    # print(f"A_new shape: {A_new.shape}")

    f = np.zeros(A_new.shape[1])


    b_eq = np.zeros(A_new.shape[0])

    # bound setting
    lb_lambda = -np.inf * np.ones(count_lambda)
    lb_gamma = np.full(count_gamma, 1e-6) # linprog에서는 이상만 먹혀서
    lb = np.concatenate([lb_lambda,lb_gamma])

    ub_lambda = np.inf * np.ones(count_lambda)
    ub_gamma = np.inf * np.ones(count_gamma)
    ub = np.concatenate([ub_lambda,ub_gamma])

    bounds = list(zip(lb,ub))

    # linprog
    # eq = A_eq*x == b_eq
    result = linprog(f,A_ub=None,b_ub=None, A_eq=A_new,b_eq=b_eq,bounds=bounds)

    if result.success:
        print("feasible solution found")
        print(f"number of bars: {count_lambda}, number of cables: {count_gamma}")
        print(result.x)

    else:
        print('Infeasible')
        print(f"number of bars: {count_lambda}, number of cables: {count_gamma}")

# # make a graph 
# G = nx.dodecahedral_graph()
# pos = nx.kamada_kawai_layout(G)
# nx.set_node_attributes(G,pos,'pos')

# planer tensegrity
G = nx.Graph()
G.add_node(0, pos=(1,0))
G.add_node(1, pos=(0,1))
G.add_node(2, pos=(-1,0))
G.add_node(3, pos=(0,-1))

# G.add_edges_from([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
# G.add_edges_from([(0,2),(1,3),(0,1),(0,3),(1,2),(2,3)])
edge_list = [(0,2), (1,3), (0,1), (0,3), (1,2), (2,3)]
G.add_edges_from(edge_list)
pos = nx.get_node_attributes(G, 'pos')

# N Matrix
num_nodes = len(pos)
N = np.zeros((2,num_nodes))
for i, (node,coords) in enumerate(pos.items()):
    N[0,i] = coords[0]
    N[1,i] = coords[1]

# C Matrix
num_members = len(G.edges())
C = np.zeros((num_members,num_nodes))
# for i, (u,v) in enumerate(G.edges()):
for i, (u,v) in enumerate(edge_list):

    C[i,u] = 1
    C[i,v] = -1

# M Matrix
M = N @ C.T

# 실제 계산 부분
for i in range(1,num_members+1):
    Sigma,count_lambda,count_gamma = get_Sigma(i,num_members)
    LP_prog_feasible(M,C,Sigma,count_lambda,count_gamma)

# print('---------------N-------------------------')
# print(N)
# print('---------------M-------------------------')
# print(M)
# print('---------------C-------------------------')
# print(C)

# print(G.edges())

