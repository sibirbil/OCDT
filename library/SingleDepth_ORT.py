# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:59:23 2024

@author: ht
"""

import numpy as np
import scipy.stats as stats
import time
from gurobipy import Model,GRB,quicksum
import gurobipy as gp
import math
import pandas as pd


def SingleDepthMIP(X,Y,Nmin,K,problem,initial_solution,Tmax=None):
    T = 3
    TB = [i for i in range(1,math.floor(T/2)+1)]
    TL = [i for i in range(math.floor(T/2)+1,T+1)]
    Lt = {j: [int(j/2**i) for i in range(int(math.log(j,2)),0,-1) if int(j/2**(i-1))%2==0] for j in range(2,T+1)}
    Rt = {j: [int(j/2**i) for i in range(int(math.log(j,2)),0,-1) if int(j/2**(i-1))%2==1] for j in range(2,T+1)}
    pt = {j: int(j/2) for j in range(2,T+1)}
    Mf = sum(Y[Y.columns[k]].max()-Y[Y.columns[k]].min() for k in range(-K,0))
    temp = (Y[[Y.columns[k] for k in range(-K,0)]] - 
            Y[[Y.columns[k] for k in range(-K,0)]].mean())**2
    Lhat = temp.mean().sum() 
    xsorted = {j: X.sort_values(by=[X.columns[j-1]],ascending=True)[X.columns[j-1]].to_numpy() for j in range(1,len(X.columns)+1)}
    
    eps = {j: min([xsorted[j][i+1]-xsorted[j][i] for i in range(len(xsorted[j])-1) 
            if xsorted[j][i+1]-xsorted[j][i] > 1e-5]) for j in range(1,len(X.columns)+1)
                if len(set(xsorted[j])) > 1}
    
    if len(eps.values()) < 1:
        return 1,1,None

    flag = False
    for j in range(1,len(X.columns)+1):
        unique_vals, counts = np.unique(xsorted[j], return_counts=True)
        cumulative_counts = np.cumsum(counts)
        for idx, count in enumerate(cumulative_counts):
            num_smaller_or_equal = count
            num_greater = len(xsorted[j]) - cumulative_counts[idx]
            if num_smaller_or_equal >= Nmin and num_greater >= Nmin:
                flag = True
                break
        if flag == True:
            break
    if flag == False:
        return 1,1,None,1e20,1e20,1e20
    
    epsmin = min(eps.values())
    epsmax = max(eps.values())
    
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    m = gp.Model("mip1", env=env)
    
    n,p = X.shape
    x = X.to_numpy()
    y = Y.to_numpy()
            
    f,B,z,a,b,d,l = {},{},{},{},{},{},{}
    
    for i in range(1,n+1):
        z[i] = m.addVar(vtype='B')
        for k in range(1,K+1):
            f[i,k] = m.addVar(vtype='C',lb=0, name="yhat"+str(i)+','+str(k))

    for t in TL:
        l[t] = m.addVar(vtype='B')
        for k in range(1,K+1):
            B[t,k] = m.addVar(vtype='C',lb=-GRB.INFINITY)
    
    for t in TB:
        b[t] = m.addVar(vtype='C',ub=1)
        d[t] = m.addVar(vtype='B')
        for j in range(1,p+1):
            a[j,t] = m.addVar(vtype='B')
    
    m.modelSense = GRB.MINIMIZE
    m.update()
    
    # Original objective function in Bertsimas and Dunn's paper
    # m.setObjective((quicksum((f[i,k]-y[i-1,k-1])**2 for k in range(1,K+1) for i in range(1,n+1))/Lhat))
    
    # Objective function only includes loss values
    m.setObjective((quicksum((f[i,k]-y[i-1,k-1])**2 for k in range(1,K+1) for i in range(1,n+1))/n))
    
    for i in range(1,n+1):
        for k in range(1,K+1):
            m.addConstr(f[i,k] - B[3,k] >= -Mf * (1-z[i]))
            m.addConstr(f[i,k] - B[3,k] <= Mf * (1-z[i]))
            m.addConstr(f[i,k] - B[2,k] >= -Mf * z[i])
            m.addConstr(f[i,k] - B[2,k] <= Mf * z[i])
    
    for i in range(1,n+1):
        m.addConstr(quicksum(a[j,1]*(x[i-1,j-1]) for j in range(1,p+1))+epsmin <= b[1] + (1+epsmax)*z[i])
        m.addConstr(quicksum(a[j,1]*x[i-1,j-1] for j in range(1,p+1)) >= b[1] - (1+epsmax)*(1-z[i]))

    
    m.addConstr(quicksum(z[i] for i in range(1,n+1)) >= Nmin)
    m.addConstr(quicksum(z[i] for i in range(1,n+1)) <= (n - Nmin))
    
    for t in TB:
        m.addConstr(quicksum(a[j,t] for j in range(1,p+1)) == 1)
    
    if problem == 'scores':
        ind = {}
        for k in range(1,K+1):
            ind[k] = m.addVar(vtype='B')
        for i in range(1,n+1):
            for k in range(1,K+1):
                m.addConstr(f[i,k] <= 1.1*ind[k])
            m.addConstr(f[i,2] >= 0.5*ind[3])
            m.addConstr(f[i,2] + f[i,3] >= 1.1*ind[1])
    elif problem == 'class':
        ind = {}
        for t in TL:
            for k in range(1,K+1):
                ind[t,k] = m.addVar(vtype='B')
            m.addConstr(quicksum(ind[t,i] for i in range(1,K+1)) <= 1)
        for t in TL:
            for k in range(1,K+1):
                m.addConstr(B[t,k] <= 1.1*ind[t,k])
    elif problem == 'synthetic_manifold':
        for t in TL:
            m.addConstr(quicksum(B[t,j+1] for j in range((K//2))) == 1)
            j=0
            for i in range((K//2),K,2):
                m.addConstr(B[t,i]-B[t,i+1] == (j+1)/10, "y_dif_constraint_"+str(j))
                j+=1
    elif problem == 'hts':
        binary_vars = {}
        for t in TL:
            for j in range(K):
                binary_vars[t,j+1] = m.addVar(vtype='B')
            for j in range(K):
                m.addConstr(B[t,j+1] <= 100*binary_vars[t,j+1], f"max_demand_{i}")
            m.addConstr(quicksum(binary_vars[t,j+1] for j in range(K)) <= 13-9, "one_prediction_constraint")
            m.addConstr(quicksum(B[t,j+1] for j in range(K)) == 15)
    else:
        raise ValueError('Single-depth MIP error: Problem is not properly defined')
        
    if Tmax:
        m.params.TimeLImit = Tmax
    
    # m.params.OutputFlag = 0
    
    if sum(initial_solution) > 0:
        m.update()
        wsloss= (sum((initial_solution[k-1]-y[i-1,k-1])**2 for k in range(1,K+1) for i in range(1,n+1))/n)
        for idx, v in enumerate([v for v in m.getVars() if v.VarName.startswith('yhat')]):
            v.Start = initial_solution[idx%K]
    m.optimize()
    
    # The expression, m.ObjVal/K, is added so as to make objective value is inline with the  
    # mean_squared_error function of sklearn
    statistics = [m.ObjVal/K,m.ObjBound,m.RunTime,int(m.NodeCount)]
    
    tot = {t:0 for t in TL}
    count= {t:0 for t in TL}
    for i in range(1,n+1):
        if z[i].x > 0.5:
            tot[3]+= sum((f[i,k].x-y[i-1,k-1])**2 for k in range(1,K+1))
            count[3]+=1
        else:
            tot[2] += sum((f[i,k].x-y[i-1,k-1])**2 for k in range(1,K+1))
            count[2]+=1
            
    mse1 = tot[2] / count[2]
    mse2 = tot[3] / count[3]
    
    predictions = []
    split_feature = None
    for t in TB:
        split_value = b[t].x
        for j in range(1,p+1):
            if a[j,t].x > 0.5:
                split_feature = j
        if split_feature:
            if split_value > X[X.columns[split_feature-1]].max() or split_value < X[X.columns[split_feature-1]].min():
                split_feature = None
                split_value = None
    for t in TL:
        predictions.append([B[t,k].x for k in range(1,K+1)])
    
    if split_feature:
        temp = X.sort_values(by=[X.columns[split_feature-1]],ascending=True)
        temp = temp.reset_index()
        split_value = temp[temp[X.columns[split_feature-1]] < split_value][X.columns[split_feature-1]].iloc[-1]
    else:
        split_feature = None
        split_value = None

    if split_feature:
        zlist = [(1-z[i].x) for i in range(1,n+1)]
        z_series = pd.Series(zlist, index=X.index)
        col_name = X.columns[split_feature-1]
        col_values = X.loc[z_series > 0.5, col_name]
        split_value = col_values.max()

    
    a = [split_feature,split_value,predictions,mse1,mse2,m.ObjVal]
    
    return split_feature,split_value,predictions,mse1,mse2,m.ObjVal


