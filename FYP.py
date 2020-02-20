import numpy as np
from numpy import linalg as LA
import pandas as pd
import yfinance as yf
import time
import networkx as nx
from matplotlib import pyplot as plt
from numpy import random as rand
import gurobipy as gu
import logging

def get_stationary_solution(g,x,y, L, eps,k):
    # randomly initialize b0
    t0 = time.time()
    b0 = np.zeros((x.size, x.size))
    #get random j <= k
    j = np.random.randint(low = 1, high = k+1)
   # print("k: {}, choice of j: {}".format(k,j))
    #get positions of j number of non zero elements
    positions = np.random.choice(x.size**2,j,replace = False)
    indexes = [(int(pos/x.size),pos%x.size) for pos in positions]
    for i, j in indexes:
        #TODO: assumption is that the values of the network follows normal
        b0[i][j] = round(np.random.normal(), 3)
    b0 = b0.reshape(x.size, x.size)

    #do first
    b_curr = b0
    t1 = time.time()
    #print("time to initiate b0 : {}".format(t1-t0))
    c = b_curr - (1 / L) * delta_g(b_curr, y, x)
    #print("g for b_curr: {}".format(g(b_curr, y, x)))
    b_next = get_top_k(c, k)
    #b_next = get_top_k_ones(c,k)

    #loop
    iter = 0
    for i in range(1,1000):
        if (g(b_curr,y,x) - g(b_next,y,x) <= eps):
            iter= i
            #print("Number of iterations: {}".format(iter))
            #print("Least squares: {}".format(g(b_next, y, x)))
            return b_next, g(b_next,y,x)
        else:
            b_curr = b_next
            t0 = time.time()
            c = b_curr - (1 / L) * delta_g(b_curr, y, x)
            t1 = time.time()
            #print('time to calulate c = {}'.format(t1-t0))
            t0 = time.time()
            b_next = get_top_k(c,k)
            #b_next = get_top_k_ones(c,k)
            t1 = time.time()
            #print('time to get top k = {}'.format(t1-t0))
            iter = i
            #print("b next: \n{}".format(b_next))
    #print("Number of iterations: {}".format(iter))
    return b_next, g(b_next,y,x)

def g(b,y,x):
    t0 = time.time()
    out = 1/2*(LA.norm(y-np.dot(b, x), ord=None)**2)
    t1 = time.time()
    #print('time to compute g() = {}'.format(t1-t0))
    return out

def delta_g(b,y,x):
    t0 = time.time()
    out = np.dot(np.dot(b,x)-y,x.transpose())
    t1 = time.time()
    #print('time to compute delta_g() = {}'.format(t1-t0))
    return out

def get_top_k(array,k):
    shape = array.shape
    array = array.flatten()
    #get index of non k largest index
    ind = np.argsort(np.absolute(array))[:len(array)-k]
    array[ind] = 0
    #print("new array: {}".format(array))
    array = array.reshape(shape)
    return array

def get_top_k_ones(array,k):
    shape = array.shape
    array = array.flatten()
    length = len(array)
    ind = np.argsort(array)[:len(array) - k]
    new_array = np.ones(length)
    new_array[ind] = 0
    res = new_array.reshape(shape)
    return res


def get_set_of_top_k(array,k):
    shape = array.shape
    array = array.flatten()
    # get index of non k largest index
    res = []
    for i in range(1,k+1):
        ind = np.argsort(np.absolute(array))[:len(array) - i]
        array[ind] = 0
        res.append(array.reshape(shape))
    return res

def get_l(x):
    w,v = LA.eig(np.dot(x.transpose(),x))
    return max(w)
def get_l_matrix(x):
    w,v = LA.eig(np.dot(x,x.transpose()))
    return max(w)

def retrieve_finance_data(stocks):
    seperator = ' '
    stocks = seperator.join(stocks)
    tickers = yf.Tickers(stocks)
    data_total = yf.download(stocks,start='2016-01-01', end = '2016-12-31')
    data = data_total['Adj Close']
    data = data.drop(data.columns[data.apply(lambda col: col.isnull().sum() > 0)], axis=1)
    #data.to_csv('./stocks_prices.csv')
    return data

def download_finance_data(stocks):
    data = retrieve_finance_data(stocks[:200])
    multiplier = len(stocks)//200
    for i in range(1, multiplier - 1):
        data1 = retrieve_finance_data(stocks[200 * i:200 * (i + 1)])
        data = data.join(data1)
    data1 = retrieve_finance_data(stocks[200 * multiplier:])
    data = data.join(data1)
    data.to_csv('./stock_prices.csv')

    #data details
    print(data.info())
    print('is na: {}'.format(data.isna().sum()))
    print('is null: {}'.format(data.isnull().sum()))
    print(data.head())
    print(data.describe())

def get_stocks(file_path):
    stocks = []
    with open(file_path,'r') as file:
        for line in file:
            sline = line.split('|')
            stocks.append(sline[0])
        file.close()
    return stocks[1:-1]

def read_data(path):
    df = pd.read_csv(path)
    return df

def standardize_data(data):
    from sklearn.preprocessing import StandardScaler
    date = data.iloc[:,0]
    data = data.drop(columns = 'Date')
    col = list(data.columns)
    data = data.to_numpy()
    scaler = StandardScaler()
    scaler.fit(data)
    data1 = scaler.transform(data)
    data = pd.DataFrame(data=data1, columns=col)
    data['Date'] = date
    return data, scaler

def loop_to_get_first_order_stationary_solution(num_of_times,g,x,y, L, eps,k):
    list_of_b = {}
    for  i in range(num_of_times):
       #print('In loop :{}'.format(i))
        b_res, g_res = get_stationary_solution(g,x,y, L, eps,k)
        list_of_b[g_res] = b_res
        '''
        for i,j in list_of_b:
            print("SSE: {}, number of non zero elements: {}".format(i,np.count_nonzero(j)))
        '''
    return min(list_of_b), list_of_b[min(list_of_b)]


def visualize(adj, data):
    G = nx.Graph()
    col = data.columns
    col_old = []
    for i in col:
        col_old.append(i+"_T-1")
    G.add_nodes_from(col,bipartite = 0)
    G.add_nodes_from(col_old,bipartite=1)
    edges = []
    for i in range (0,len(adj[0])):
        for j in range(0,len(adj[0])):
            if (abs(adj[i][j])>0):
                edges.append([col[i],col_old[j]])
    G.add_edges_from(edges)
    nx.draw(G)
    plt.show()

def visualize_network(adj, col, col_old):
    G = nx.Graph()
    G.add_nodes_from(col)
    G.add_nodes_from(col_old)
    old_pos = {j:(-1,i) for i,j in enumerate(col_old)}
    new_pos = {j:(1,i) for i,j in enumerate(col)}
    old_pos.update(new_pos)
    initial_pos = old_pos
    temp1 = ','.join(col_old)
    temp2 = ',' + ','.join(col)
    temp1 += temp2
    nodes_to_be_fixed = temp1.split(',')
    colour_map = ['skyblue' for i in range(0,len(col))] + ['pink' for j in range(0,len(col_old))]
    edges = []
    for i in range (0,len(adj[0])):
        for j in range(0,len(adj[0])):
            if (abs(adj[i][j])>0):
                edges.append([col[i],col_old[j]])
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, k=3, pos = initial_pos, fixed = nodes_to_be_fixed, iterations=100)
    #pos=nx.fruchterman_reingold_layout(G,k = 3,iterations = 500)
    #pos = nx.nx_pydot.graphviz_layout(G)
    nx.draw(G, node_color=colour_map, with_labels=True, font_weight='bold',font_size = 6, width=1, node_size=90, pos=pos)

#input: y and x are col vectors
def optimize_MIO(y,x,Mu,Ml,Muz,Mlz,k):
    m = gu.Model("mip1")
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 500)
    m.setParam('MIPGapAbs',0.01)
    #convert to row vectors
    x = x.reshape(1,-1).flatten()
    y = y.reshape(1,-1).flatten()
    num = len(y)
    beta = m.addVars([(i,j) for i in range(num) for j in range(num)], vtype = gu.GRB.CONTINUOUS, name ='b')
    zeta = m.addVars([(i, j) for i in range(num) for j in range(num)], vtype=gu.GRB.BINARY, name ='z')
    b_linf = m.addVar(vtype=gu.GRB.CONTINUOUS, name = 'b_linf')
    xb_linf = m.addVar(vtype=gu.GRB.CONTINUOUS, name='xb_linf')
    m.update()
    xbeta = []
    obj = []
    #b_0 = beta.select(0,0)[0]
    for z in range(num):
        bx = []
        #for each row, get the dot product of row z of beta with x
        for i,j in zip(beta.select(z,'*'),x):
            bx.append(i*j)
        obj.append((y[z] - sum(bx)) * (y[z] - sum(bx)))
        #add variable for bx for constraints
        bx_var = m.addVar(vtype=gu.GRB.CONTINUOUS, name='bx{}'.format(z))
        m.addConstr(bx_var == sum(bx),'cbx{}'.format(z))
        xbeta.append(bx_var)
        #make beta for diagonal the same
        #b_curr = beta.select(z,z)[0]
        #m.addConstr(b_curr, gu.GRB.EQUAL, b_0, 'diag{}'.format(z))
    m.update()
    obj = 1/2 * sum(obj)
    m.setObjective(obj)
    #set zeta to be indicator variables
    for i,j in zip(beta.select(),zeta.select()):
        m.addConstr((j == 0) >> (i == 0.0))
    #assign variables for l_inf norm
    m.addGenConstrMax(b_linf, beta.select())
    m.addGenConstrMax(xb_linf,xbeta)
    c1 = b_linf <= Mu
    c2a = beta.sum() <= Ml
    c2b = beta.sum() >= -Ml
    c3 = xb_linf <= Muz
    c4a = sum(xbeta) <= Mlz
    c4b = sum(xbeta) >= -Mlz
    c5 = zeta.sum() <= k
    m.addConstr(c1)
    m.addConstr(c2a)
    m.addConstr(c2b)
    m.addConstr(c3)
    m.addConstr(c4a)
    m.addConstr(c4b)
    m.addConstr(c5)
    m.update()
    m.write('MIO.lp')
    m.optimize()
    return m

def optimize_MIO_diag(y,x,Mu,Ml,Muz,Mlz,k):
    m = gu.Model("mip1")
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 500)
    m.setParam('MIPGapAbs', 0.01)
    #convert to row vectors
    x = x.reshape(1,-1).flatten()
    y = y.reshape(1,-1).flatten()
    num = len(y)
    beta = m.addVars([(i,j) for i in range(num) for j in range(num)], vtype = gu.GRB.CONTINUOUS, name ='b')
    zeta = m.addVars([(i, j) for i in range(num) for j in range(num)], vtype=gu.GRB.BINARY, name ='z')
    abs_beta = m.addVars([(i,j) for i in range(num) for j in range(num)], vtype = gu.GRB.CONTINUOUS, name ='ab')
    b_linf = m.addVar(vtype=gu.GRB.CONTINUOUS, name = 'b_linf')
    xb_linf = m.addVar(vtype=gu.GRB.CONTINUOUS, name='xb_linf')
    m.update()
    xbeta = []
    obj = []
    diag_indicator = []
    b_0 = beta.select(0,0)[0]
    for z in range(num):
        bx = []
        #for each row, get the dot product of row z of beta with x
        for i,j in zip(beta.select(z,'*'),x):
            bx.append(i*j)
        obj.append((y[z] - sum(bx)) * (y[z] - sum(bx)))
        #add variable for bx for constraints
        bx_var = m.addVar(vtype=gu.GRB.CONTINUOUS, name='bx{}'.format(z))
        m.addConstr(bx_var == sum(bx),'cbx{}'.format(z))
        xbeta.append(bx_var)
        #make beta for diagonal the same (skip 0)
        if (z>0):
            b_curr = beta.select(z,z)[0]
            m.addConstr(b_curr, gu.GRB.EQUAL, b_0, 'diag{}'.format(z))
        #get indicator for diag values
        diag_indicator.append(zeta.select(z,z)[0])
    m.update()
    obj = 1/2 * sum(obj)
    m.setObjective(obj)
    #set zeta to be indicator variables
    for i,j in zip(abs_beta.select(),zeta.select()):
        m.addGenConstrIndicator(j,False,i,gu.GRB.EQUAL,0)
        #m.addConstr(i<=Mu*j)
        #m.addConstr(-Mu*j<=i)
    #set abs_beta to be abs variables for beta
    for i,j in zip(beta.select(),abs_beta.select()):
        m.addGenConstrAbs(j,i)
    #assign variables for l_inf norm
    m.addGenConstrMax(b_linf, beta.select())
    m.addGenConstrMax(xb_linf,xbeta)
    #for each absolute row sum of beta set constraint
    #for each absolute column sum of beta set constraint
    for i in range(num):
        #row constraint
        c1 = sum(abs_beta.select(i,'*')) <= Mu
        #column constraint
        c2 = sum(abs_beta.select('*',i)) <= Ml
        m.addConstr(c1)
        m.addConstr(c2)
    c3 = xb_linf <= Muz
    c4a = sum(xbeta) <= Mlz
    c4b = sum(xbeta) >= -Mlz
    c5 = zeta.sum() <= k + num
    #to add constraint to ensure the diag values are not zero
    c6 = sum(diag_indicator) == num
    m.addConstr(c1)
    m.addConstr(c3)
    m.addConstr(c4a)
    m.addConstr(c4b)
    m.addConstr(c5)
    m.addConstr(c6)
    m.update()
    m.write('MIO.lp')
    m.optimize()
    return m


def retrieve_optimized_results_MIO(m,p):
    vars = m.getVars()
    print('beta1:{}'.format(vars[0].x))
    A = np.zeros(shape=(p,p))
    Z = np.zeros(shape=(p,p))
    for i in range(p):
        for j in range(p):
            A[i][j] = vars[p * i + j].x
            Z[i][j] = vars[(p**2)+(p * i + j)].x
    obj = m.objVal
    return A,Z,obj


def retrieve_optimized_results_lasso(m,p):
    vars = m.getVars()
    p2 = p**2
    A = np.zeros(shape=(p,p))
    #D = np.zeros(shape=(p,p))
    for i in range(p):
        for j in range(p):
            A[i][j] = vars[p * i + j].x
            #D[i][j] = vars[p2 + p* i + j].x
    '''
    for i in m.getVars():
        print("{}: {}".format(i.varName, i.x))
    print("obj val: {}".format(m.objVal))
    '''
    obj = m.objVal
    return A,obj

def estimate_Mu(b, tau=2):
    b = abs(b.flatten())
    ind = np.argsort(b)[-1]
    return tau * b[ind]
def estimate_Mu_infty(b,tau=2):
    N = b.shape[0]
    b = abs(b)
    rows = []
    for i in range(N):
        rows.append(sum(b[i,:]))
    return max(rows)
def get_u(X):
    #X is a n*p matrix, where p is the number of random variables
    X = X.reshape(1,-1)
    pairs = [abs(X[0][i] * X[0][j]) for i in range (X.shape[1]) for j in range (i,X.shape[1])]
    return max(pairs)

def estimate_Ml_new(Mu,k):
    return k*Mu

def estimate_Ml_matrix_norm(X,Y):
    N = X.shape[0]
    yx = abs(np.dot(Y, X.transpose()))
    yx_col = []
    for j in range(N):
        yx_col.append(sum(yx[:,j]))
    first = max(yx_col)
    square = np.linalg.pinv(abs(np.dot(X, X.transpose())))
    square_col = []
    for j in range(N):
        square_col.append(sum(square[:,j]))
    second = max(square_col)
    return first * second

def estimate_Ml(X,Y,k):
    yx = abs(np.dot(Y,X.transpose())).flatten()
    ind = np.argsort(yx)
    first = sum(yx[ind][len(yx)-k:])
    square = np.linalg.pinv(abs(np.dot(X,X.transpose()))).flatten()
    ind = np.argsort(square)
    second = sum(square[ind][len(square)-k:])
    return first * second

def estimate_Mlz(X,Mu):
    p = X.shape[0]
    x = sum(abs(X.flatten()))
    return p * Mu * x

def estimate_MuZ(X,Mu):
    x = sum(abs(X.flatten()))
    return Mu * x

#returns Y_t, Y_(t-1), A
def generate_dyad_simulation_data(N,T):
    b0,b1,b2 = 0,0.2,0.5
    choices = [(1,1),(0,1),(1,0),(0,0)]
    weights = [0.1,N**(-0.8)]
    cum_weights = [0] + list(np.cumsum(weights)) + [1]
    print("in dyad simulation: cum_weights: {}".format(cum_weights))
    A = np.zeros(shape =(N,N))
    for z in range (1,N+1):
        i = rand.randint(0,N)
        j = rand.randint(0,N)
        if (i==j):
            continue
        r = rand.random()
        if (0<r<cum_weights[1]):
            c = choices[0]
            A[i][j] = c[0]
            A[j][i] = c[1]
        elif (cum_weights[2]<r<cum_weights[3]):
            c = choices[3]
            A[i][j] = c[0]
            A[j][i] = c[1]
        elif (cum_weights[1]<r<cum_weights[2]):
            c = choices[int(round(rand.random(),0)) + 1]
            A[i][j] = c[0]
            A[j][i] = c[1]
    '''
    n = [LA.norm(A[i], ord = 2)**(-1/2) for i in range(A.shape[0])]
    W = np.dot(np.diag(n),A)
    G = b1*W + b2*np.eye(N)
    I = np.eye(N)
    miu = (I -G)**(-1)*b0
    #sigma = (I - np.kron(G,G))**(-1)
    print(miu)
    #print(sigma)
    '''
    y0 = rand.randn(N)
    e0 = rand.randn(N)
    n_inv = np.zeros(N)
    for i in range(N):
        out_deg = sum(A[i,j] for j in range(N))
        n_inv[i] = 1/out_deg if out_deg > 0 else 0
    y1 = np.zeros(N)
    for i in range(N):
        y1[i] = b1*n_inv[i]*sum(A[i,j]*y0[i] for j in range(N)) + b2*y0[i]
    #convert to column vector
    y0 = y0.reshape(-1,1)
    y1 = y1.reshape(-1,1)
    return A,y0,y1

def solve_MIO(start,stop,X,Y,A):
    logger = logging.getLogger('root')
    logger.info('Running MIO for {} nodes'.format(len(X.flatten())))
    logger.info('Parameters:')
    #logger.info('K from {} to {}'.format(start,stop-1))
    t0 = time.time()
    res = {}
    obj_list = []
    for k in range(start,stop):
        t00 = time.time()
        #get estimates
        g_res, b_res = loop_to_get_first_order_stationary_solution(50,g,X,Y, get_l_matrix(X), 1,k)
        #print("after first order: b:\n{}".format(b_res))
        Mu = estimate_Mu(b_res,2)
        Ml = estimate_Ml_new(Mu,k)
        Muz = estimate_MuZ(X,Mu)
        Mlz = estimate_Mlz(X,Mu)
        m = optimize_MIO(Y,X,Mu,Ml,Muz,Mlz,k)
        a, z, o = retrieve_optimized_results_MIO(m, len(X.flatten()))
        res[k] = (o, a, z)
        obj_list.append(o)
        t01 = time.time()
        logger.info("K: {}, Mu: {}, Ml:{}, Muz:{}, Mlz:{}, runtime: {}, obj: {}".format(k, Mu, Ml, Muz, Mlz,t01-t00,o))
    t1 = time.time()
    runtime = t1-t0
    #get optimized result
    min_ind = np.argsort(obj_list)[0] + start
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y = [(res[i][0]) for i in res]
    lowest_k = np.argsort(y)[0] + start
    x = np.arange(start,stop)
    p= X.shape[0]
    ax.plot(x,y)
    ax.axvline(x = lowest_k,ls= '--',linewidth = 1)
    ax.set_xlabel('k')
    ax.set_ylabel('Objective Value')
    ax.set_title('Objective value of optimal Adjacency matrix against k')
    ax.text(0.55,0.5,"Minimum Objective Value:\n{}".format(obj_list[min_ind-start]),va='center',ha='center',transform=ax.transAxes)
    plt.savefig('./plots/obj_{}'.format(p))
    plt.close(fig)
    min_obj = obj_list[min_ind-start]
    predicted_error = min_obj / sum([i ** 2 for i in Y])[0]
    estimated_matrix = res[min_ind][1]
    indicator_matrix = res[min_ind][2]
    handle_results('MIO', runtime, min_obj,predicted_error, estimated_matrix, A, lowest_k, indicator_matrix = indicator_matrix)
    return estimated_matrix

def solve_MIO_diag(start,stop,X,Y,A):
    logger = logging.getLogger('root')
    logger.info('Running MIO for {} nodes'.format(len(X.flatten())))
    logger.info('Parameters:')
    #logger.info('K from {} to {}'.format(start,stop-1))
    t0 = time.time()
    res = {}
    obj_list = []
    for k in range(start,stop):
        t00 = time.time()
        #get estimates
        t_s_start = time.time()
        g_res, b_res = loop_to_get_first_order_stationary_solution(50,g,X,Y, get_l_matrix(X), 0.1,k)
        t_s_stop=time.time()
        logger.info('Time to get stationary_solutiuon:{}, with SSE:{}'.format(t_s_stop-t_s_start,g_res))
        #print("after first order: b:\n{}".format(b_res))
        #Mu = estimate_Mu(b_res)
        Mu = estimate_Mu_infty(b_res)
        Ml = estimate_Ml_matrix_norm(X,Y)
        Ml_new = estimate_Ml_new(Mu,k)
        Muz = estimate_MuZ(X,Mu)
        Mlz = estimate_Mlz(X,Mu)
        logger.info('kMu:{}'.format(estimate_Ml_new(Mu,k)))
        logger.info('Ml:{}'.format(Ml))
        m = optimize_MIO_diag(Y,X,Mu,Ml,Muz,Mlz,k)
        a, z, o = retrieve_optimized_results_MIO(m, len(X.flatten()))
        res[k] = (o, a, z)
        obj_list.append(o)
        t01 = time.time()
        logger.info("K: {}, Mu: {}, Ml:{}, Muz:{}, Mlz:{}, runtime: {}, obj: {}: # of non-zero:{}".format(k, Mu, Ml, Muz, Mlz,t01-t00,o,np.count_nonzero(a)))
    t1 = time.time()
    runtime = t1-t0
    #get optimized result
    min_ind = np.argsort(obj_list)[0] + start
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y = [(res[i][0]) for i in res]
    lowest_k = np.argsort(y)[0] + start
    x = np.arange(start,stop)
    p= X.shape[0]
    ax.plot(x,y)
    ax.axvline(x = lowest_k,ls= '--',linewidth = 1)
    ax.set_xlabel('k')
    ax.set_ylabel('Objective Value')
    ax.set_title('Objective value of optimal Adjacency matrix against k')
    ax.text(0.55,0.5,"Minimum Objective Value:\n{}".format(obj_list[min_ind-start]),va='center',ha='center',transform=ax.transAxes)
    plt.savefig('./plots/obj_{}'.format(p))
    plt.close(fig)
    min_obj = obj_list[min_ind-start]
    #Get Prediction error min_obj/|Y|^2
    predicted_error = min_obj/sum([i**2 for i in Y])[0]
    estimated_matrix = res[min_ind][1]
    indicator_matrix = res[min_ind][2]
    handle_results('MIO_diag', runtime, min_obj,predicted_error, estimated_matrix, A, lowest_k, indicator_matrix = indicator_matrix)
    return estimated_matrix

def handle_results(model,runtime,min_obj,predicted_error,estimated_matrix, actual_matrix,lowest_k = None, alpha = None, indicator_matrix= None, save=True):
    logger = logging.getLogger('root')
    nodes = actual_matrix.shape[0]
    #get diag values
    diag = []
    for i in range(estimated_matrix.shape[0]):
        diag.append(estimated_matrix[i,i])
        estimated_matrix[i, i] = 0
    print(diag)
    non_zero_estimated = np.count_nonzero(estimated_matrix)
    non_zero_actual = np.count_nonzero(actual_matrix)
    logger.info("Runtime: {}".format(runtime))
    logger.info("Minimum Objective Value: {}".format(min_obj))
    logger.info("Predicted Error: {}".format(float(predicted_error)))
    logger.info("Number of non zeroes in estimated matrix: {}".format(non_zero_estimated))
    logger.info("Number of non zeroes in actual matrix: {}".format(non_zero_actual))
    print("obj: {}".format(min_obj))
    print("Optimized Adjacency matrix:\n{}".format(estimated_matrix))
    print("Actual Adjacency matrix\n{}".format(actual_matrix))
    if (indicator_matrix is not None):
        non_zero_indicator = np.count_nonzero(indicator_matrix)
        print("Optimized Indicator matrix:\n{}".format(indicator_matrix))
        print(
            "# of non zeros in estimated matrix:{}\n# of non zeros in indicator matrix:{}\n# of non zeros in actual matrix:{}".format(
                non_zero_estimated, np.count_nonzero(indicator_matrix), non_zero_actual))
    else:
        print(
            "# of non zeros in estimated matrix:{}\n# of non zeros in actual matrix:{}".format(
                non_zero_estimated, non_zero_actual))
    if (save):
        from pathlib import Path
        file = Path('./results.csv')
        if (not file.is_file()):
            df = pd.DataFrame(columns=['Model', 'Nodes', 'Runtime', 'Minimum Objective Value','Predicted Error', 'Non-zero estimated','Non-zero actual','Lowest_k','Alpha'])
            df.to_csv('./results.csv',index=False)

        df = pd.read_csv('./results.csv')
        df = df.append({'Model':model, 'Nodes': nodes, 'Runtime': runtime, 'Minimum Objective Value': min_obj,'Predicted Error':predicted_error, 'Non-zero estimated': non_zero_estimated,'Non-zero actual': non_zero_actual,'Lowest_k': lowest_k,'Alpha':alpha},ignore_index=True)
        df.to_csv('./results.csv',index=False)

def optimize_lasso(x,y,alpha):
    m = gu.Model("qp")
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 500)
    # convert to row vectors
    x = x.reshape(1, -1).flatten()
    y = y.reshape(1, -1).flatten()
    num = len(y)
    beta = m.addVars([(i, j) for i in range(num) for j in range(num)], vtype=gu.GRB.CONTINUOUS, name='b')
    dummy = m.addVars([(i, j) for i in range(num) for j in range(num)], vtype=gu.GRB.CONTINUOUS, name='d')
    m.update()
    obj = []
    b_0 = beta.select(0,0)[0]
    for z in range(num):
        bx = []
        #for each row, get the dot product of row z of beta with x
        for i,j in zip(beta.select(z,'*'),x):
            bx.append(i*j)
        obj.append((y[z] - gu.quicksum(bx))*(y[z] - gu.quicksum(bx)))
        #print('obj: {}'.format(obj))
        #add constraint to make diagonal betas the same
        #b_curr = beta.select(z,z)[0]
        #m.addConstr(b_curr,gu.GRB.EQUAL, b_0,'diag{}'.format(z))
    for i,j in zip(dummy.select(),beta.select()):
        m.addConstr(i == gu.abs_(j))
    obj_func = 1/2 * gu.quicksum(obj) + alpha * dummy.sum()
    m.setObjective(obj_func)
    m.update()
    m.write('lasso.lp')
    m.optimize()
    return m

def save_dyad_simulation_dataset(k):
    A, y0, y1 = generate_dyad_simulation_data(k, 1)
    r = np.arange(len(y0.flatten())+2)
    B = A.transpose()
    col = []
    col.append(y0.flatten())
    col.append(y1.flatten())
    for i in range(len(B[0])):
        col.append(B[i])
    d = dict(zip(r,col))
    df = pd.DataFrame(d)
    df.to_csv('./Data/{}.csv'.format(k))

def get_data(filepath):
    df = pd.read_csv(filepath)
    y0 = np.array(df.iloc[:,1]).reshape(-1,1)
    y1 = np.array(df.iloc[:,2]).reshape(-1,1)
    A = np.array(df.iloc[:,3:])
    return y0,y1,A

def solve_lasso(X,Y,A):
    logger = logging.getLogger('root')
    logger.info('Running Lasso for {} nodes'.format(len(X.flatten())))
    alpha = 0.7
    lasso_res = []
    t0 = time.time()
    m = optimize_lasso(X,Y,alpha)
    t1 = time.time()
    runtime = t1-t0
    a, obj = retrieve_optimized_results_lasso(m, len(X.flatten()))
    predicted_error = obj/sum([i**2 for i in Y])[0]
    handle_results('Lasso', runtime, obj,predicted_error, a, A, alpha=alpha)
    return obj

def soft_thres(beta,l):
    return np.sign(beta) * np.maximum(np.abs(beta) - l, 0.)


def ista(X,Y,A,l,maxit):
    #l: regularization parameter
    L = get_l(X) * 2
    num = X.shape[0]
    beta = np.zeros((num,num))
    pobj = []
    time0 = time.time()
    for _ in range(maxit):
        beta = soft_thres(beta-2,l/L)


def test_for_k():
    a = np.arange(-12,13).reshape(5,5)
    #a = np.random.normal(size=(5,5))
    print("before: {}".format(a))
    b = get_top_k(a,10)
    print("after: {}".format(b))

def test_for_k_ones():
    a = np.arange(-12, 13).reshape(5, 5)
    # a = np.random.normal(size=(5,5))
    print("before: {}".format(a))
    b = get_top_k_ones(a, 10)
    print("after: {}".format(b))

def test_for_g():
    b = np.arange(9).reshape(3,3)
    y = np.array([1,2,3]).reshape(-1,1)
    x = np.array([4,5,6]).reshape(-1,1)
    print("shape of b: {}".format(b.shape))
    print("shape of y: {}".format(y.shape))
    print("shape of x: {}".format(x.shape))
    print(g(b,y,x))

def test_optimize_MIO():
    y = np.random.normal(size=5)
    x = np.random.rand(5)
    print("x:{} \ny:{}".format(x,y))
    Mu=Ml=Muz=Mlz = 10000.0
    k=10
    m = optimize_MIO(y,x,Mu,Ml,Muz,Mlz,k)
    vars = m.getVars()
    print(vars)
    A = np.zeros(shape=(5,5))
    for i in range(5):
        for j in range(5):
            A[i][j] = vars[5 * i + j].x
    print(A)
    for i in m.getVars():
        print("{}: {}".format(i.varName, i.x))
    print("obj val: {}".format(m.objVal))

def test_stationary_solution_for_epsilon(num_of_times, g, x, y, L, k):
    logger.info('Starting test for discrete stationary algo for {} nodes'.format(x.shape[0]))
    for eps in np.arange(0.1,1.0,0.2):
        start = time.time()
        g_res, b_res = loop_to_get_first_order_stationary_solution(num_of_times, g, x, y, L, eps, k)
        stop = time.time()
        logger.info('eps:{}, SSE:{}, Time:{}'.format(eps,g_res,stop-start))


def plot_results():
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    df = pd.read_csv('./results.csv')
    #df = df.sort_values(by=['Nodes','Model'])
    df = df[(df.Model == 'MIO')].sort_values(by=['Nodes']).reset_index()
    df1 = df[(df.Model == 'MIO')].filter(items = ['Nodes','Lowest_k'])
    x = np.array([i[0] for i in df1.to_numpy(dtype = int)]).reshape(-1,1)
    y = np.array([i[1] for i in df1.to_numpy(dtype = int)]).reshape(-1,1)
    reg = LinearRegression().fit(x,y)
    x_test = np.array([250,300,350,400,450,500]).reshape(-1,1)
    y_test = reg.predict(x_test)
    x_new = np.concatenate((x.flatten(),x_test.flatten()))
    y_new = np.concatenate((y.flatten(),y_test.flatten()))
    print(x_test[-3],y_test[-3])
    plt.plot(x_new,y_new)
    print(df1)

def get_nonzeros_for_data():
    for i in [5,10,20,50,70,100]:
        y0,y1,A = get_data('./Data/{}.csv'.format(i))
        print('Number of non-zeros in A with {} Nodes: {}'.format(i,np.count_nonzero(A.flatten())))

def plot_prediction_error():
    df = pd.read_csv('./results.csv')
    mio = df.iloc[:5,:]
    lasso = df.iloc[5:10,:]
    labels = mio.Nodes.to_numpy()
    mio_predicted_error = [round(i,2) for i in mio['Predicted Error'].to_numpy()]
    lasso_predicted_error= [round(i,2) for i in lasso['Predicted Error'].to_numpy()]
    print(mio_predicted_error)
    x = np.arange(len(labels))
    width=0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, mio_predicted_error, width, label='MIO')
    rects2 = ax.bar(x + width / 2, lasso_predicted_error, width, label='Lasso')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Prediction Error')
    ax.set_ylim(0,0.5)
    ax.set_title('Prediction Error by MIO and Lasso')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Size of Network')
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.plot()
    plt.savefig('./plots/prediction_error')
    plt.close(fig)

def plot_network(A,a,y0,y1):
    # plot fig
    y1_for = ['{}'.format(i) for i in range(1, len(y1) + 1)]
    y0_for = ["{} ".format(i) for i in range(1, len(y0) + 1)]

    fig = plt.figure()
    ax1 = plt.subplot(121)
    visualize_network(A, y1_for, y0_for)
    ax1.set_title('Actual Network')
    plt.text(0.5, 0, 'Number of non-zero edges: {}'.format(np.count_nonzero(A)), ha='center', va='bottom',
             transform=ax1.transAxes)
    ax2 = plt.subplot(122)
    visualize_network(a, y1_for, y0_for)
    ax2.set_title('Estimated Network')
    plt.text(0.5, 0, 'Number of non-zero edges: {}'.format(np.count_nonzero(a)), ha='center', va='bottom',
             transform=ax2.transAxes)
    plt.savefig('./plots/network_{}'.format(y0.shape[0]), bbox_inches='tight')
    plt.close(fig)
def plot_dataset_network(y0,y1,A):
    y1_for = ['{}'.format(i) for i in range(1, len(y1) + 1)]
    y0_for = ["{} ".format(i) for i in range(1, len(y0) + 1)]
    fig,ax  = plt.subplots()
    visualize_network(A, y1_for, y0_for)
    ax.set_title('Network of size {}'.format(y0.shape[0]))
    plt.text(0.5, 0, 'Number of non-zero edges: {}'.format(np.count_nonzero(A)), ha='center', va='bottom',
             transform=ax.transAxes)
    plt.savefig('./plots/actual_network_{}'.format(y0.shape[0]), bbox_inches='tight')


if __name__ == "__main__":
    logging.basicConfig(filename='main.log', level=logging.INFO, format = "%(asctime)s;%(message)s")
    logger = logging.getLogger('root')
    logger.info('Starting Task')

    #plot_prediction_error()
    #save_dyad_simulation_dataset(90)
    rand.seed()
    y0,y1,A = get_data('./Data/90.csv')
    #test_stationary_solution_for_epsilon(50,g,y0,y1, get_l_matrix(y0),9)
    #plot_dataset_network(y0,y1,A)
    plot_prediction_error()
    #a = solve_MIO_diag(29,34,y0,y1,A)
    #a = solve_MIO(26,31,y0,y1,A)
    #solve_lasso(y0,y1,A)
    #plot_network(A,a,y0,y1)