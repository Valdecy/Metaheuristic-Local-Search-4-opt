############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Local Search-4-opt

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Local_Search-4-opt, File: Python-MH-Local Search-4-opt.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Local_Search-4-opt>

############################################################################

# Required Libraries
import pandas as pd
import random
import numpy  as np
import copy
from matplotlib import pyplot as plt

# Function: Tour Distance
def distance_calc(Xdata, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + Xdata.iloc[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: Euclidean Distance 
def euclidean_distance(x, y):       
    distance = 0
    for j in range(0, len(x)):
        distance = (x.iloc[j] - y.iloc[j])**2 + distance   
    return distance**(1/2) 

# Function: Initial Seed
def seed_function(Xdata):
    seed = [[],float("inf")]
    sequence = random.sample(list(range(1,Xdata.shape[0]+1)), Xdata.shape[0])
    sequence.append(sequence[0])
    seed[0] = sequence
    seed[1] = distance_calc(Xdata, seed)
    return seed

# Function: Build Distance Matrix
def buid_distance_matrix(coordinates):
    Xdata = pd.DataFrame(np.zeros((coordinates.shape[0], coordinates.shape[0])))
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            if (i != j):
                x = coordinates.iloc[i,:]
                y = coordinates.iloc[j,:]
                Xdata.iloc[i,j] = euclidean_distance(x, y)        
    return Xdata

# Function: Tour Plot
def plot_tour_distance_matrix (Xdata, city_tour):
    m = Xdata.copy(deep = True)
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            m.iloc[i,j] = (1/2)*(Xdata.iloc[0,j]**2 + Xdata.iloc[i,0]**2 - Xdata.iloc[i,j]**2)    
    m = m.values
    w, u = np.linalg.eig(np.matmul(m.T, m))
    s = (np.diag(np.sort(w)[::-1]))**(1/2) 
    coordinates = np.matmul(u, s**(1/2))
    coordinates = coordinates.real[:,0:2]
    xy = pd.DataFrame(np.zeros((len(city_tour[0]), 2)))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy.iloc[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy.iloc[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy.iloc[:,0], xy.iloc[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy.iloc[0,0], xy.iloc[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy.iloc[1,0], xy.iloc[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Tour Plot
def plot_tour_coordinates (coordinates, city_tour):
    coordinates = coordinates.values
    xy = pd.DataFrame(np.zeros((len(city_tour[0]), 2)))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy.iloc[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy.iloc[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy.iloc[:,0], xy.iloc[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy.iloc[0,0], xy.iloc[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy.iloc[1,0], xy.iloc[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: 2_opt
def local_search_2_opt(Xdata, city_tour):
    city_list = copy.deepcopy(city_tour)
    best_route = copy.deepcopy(city_list)
    seed = copy.deepcopy(city_list)        
    for i in range(0, len(city_list[0]) - 2):
        for j in range(i+1, len(city_list[0]) - 1):
            best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
            best_route[0][-1]  = best_route[0][0]                          
            best_route[1] = distance_calc(Xdata, best_route)    
            if (best_route[1] < city_list[1]):
                city_list[1] = copy.deepcopy(best_route[1])
                for n in range(0, len(city_list[0])): 
                    city_list[0][n] = best_route[0][n]          
            best_route = copy.deepcopy(seed) 
    print("Best 2-opt solution found =", city_list)
    return city_list

# Function: 3_opt
def local_search_3_opt(Xdata, city_tour):
    city_list = copy.deepcopy(city_tour)
    best_route = copy.deepcopy(city_list)
    best_route_01 = [[],float("inf")]
    best_route_02 = [[],float("inf")]
    best_route_03 = [[],float("inf")]
    best_route_04 = [[],float("inf")]       
    seed = copy.deepcopy(city_list)        
    for i in range(0, len(city_list[0]) - 3):
        for j in range(i+1, len(city_list[0]) - 2):
            for k in range(j+1, len(city_list[0]) - 1): 
                best_route_01[0] = best_route[0][:i+1] + best_route[0][j+1:k+1] + best_route[0][i+1:j+1] + best_route[0][k+1:]
                best_route_01[1] = distance_calc(Xdata, best_route_01)
                best_route_02[0] = best_route[0][:i+1] + list(reversed(best_route[0][i+1:j+1])) + list(reversed(best_route[0][j+1:k+1])) + best_route[0][k+1:]
                best_route_02[1] = distance_calc(Xdata, best_route_02)
                best_route_03[0] = best_route[0][:i+1] + list(reversed(best_route[0][j+1:k+1])) + best_route[0][i+1:j+1] + best_route[0][k+1:]
                best_route_03[1] = distance_calc(Xdata, best_route_03)
                best_route_04[0] = best_route[0][:i+1] + best_route[0][j+1:k+1] + list(reversed(best_route[0][i+1:j+1])) + best_route[0][k+1:]
                best_route_04[1] = distance_calc(Xdata, best_route_04)
                       
                if(best_route_01[1]  < best_route[1]):
                    best_route[1] = copy.deepcopy(best_route_01[1])
                    for n in range(0, len(best_route[0])): 
                        best_route[0][n] = best_route_01[0][n] 
                        
                elif(best_route_02[1]  < best_route[1]):
                    best_route[1] = copy.deepcopy(best_route_02[1])
                    for n in range(0, len(best_route[0])): 
                        best_route[0][n] = best_route_02[0][n] 
                        
                elif(best_route_03[1]  < best_route[1]):
                    best_route[1] = copy.deepcopy(best_route_03[1])
                    for n in range(0, len(best_route[0])): 
                        best_route[0][n] = best_route_03[0][n]
                        
                elif(best_route_04[1]  < best_route[1]):
                    best_route[1] = copy.deepcopy(best_route_04[1])
                    for n in range(0, len(best_route[0])): 
                        best_route[0][n] = best_route_04[0][n] 
                        
            if (best_route[1] < city_list[1]):
                city_list[1] = copy.deepcopy(best_route[1])
                for n in range(0, len(city_list[0])): 
                    city_list[0][n] = best_route[0][n]              
            best_route = copy.deepcopy(seed)
    print("Best 2-opt solution found =", city_list)
    return city_list

# Function: 4_opt
def local_search_4_opt(Xdata, city_tour, recursive_seeding = 1):
    count = 0
    city_list = copy.deepcopy(city_tour)
    while (count < recursive_seeding):
        best_route   = copy.deepcopy(city_list)
        best_route_01 = local_search_2_opt(Xdata, best_route)
        best_route_02 = local_search_3_opt(Xdata, best_route)
        best_route_03 = [[],float("inf")]
        best_route_04 = [[],float("inf")]
        best_route_05 = [[],float("inf")]
        best_route_06 = [[],float("inf")]
        best_route_07 = [[],float("inf")]
        best_route_08 = [[],float("inf")] 
        best_route_09 = [[],float("inf")]
        best_route_10 = [[],float("inf")]
        best_route_11 = [[],float("inf")]
        best_route_12 = [[],float("inf")]
        best_route_13 = [[],float("inf")]
        best_route_14 = [[],float("inf")]
        best_route_15 = [[],float("inf")]
        best_route_16 = [[],float("inf")]
        best_route_17 = [[],float("inf")]
        best_route_18 = [[],float("inf")]
        best_route_19 = [[],float("inf")]
        best_route_20 = [[],float("inf")] 
        best_route_21 = [[],float("inf")]
        best_route_22 = [[],float("inf")]
        best_route_23 = [[],float("inf")]
        best_route_24 = [[],float("inf")]
        best_route_25 = [[],float("inf")]
        best_route_26 = [[],float("inf")] 
        best_route_27 = [[],float("inf")] 
        seed = copy.deepcopy(city_list)        
        for i in range(0, len(city_list[0]) - 4):
            for j in range(i+1, len(city_list[0]) - 3):
                for k in range(j+1, len(city_list[0]) - 2): 
                    for L in range(k+1, len(city_list[0]) - 1):                         
                        best_route_03[0] = best_route[0][:i+1] + best_route[0][k+1:L+1] + best_route[0][j+1:k+1] + best_route[0][i+1:j+1] + best_route[0][L+1:]
                        best_route_03[1] = distance_calc(Xdata, best_route_03)                        
                        best_route_04[0] = best_route[0][:i+1] + list(reversed(best_route[0][i+1:j+1])) + best_route[0][j+1:k+1] + list(reversed(best_route[0][k+1:L+1])) + best_route[0][L+1:]                  
                        best_route_04[1] = distance_calc(Xdata, best_route_04)                        
                        best_route_05[0] = best_route[0][:i+1] + list(reversed(best_route[0][i+1:j+1])) + list(reversed(best_route[0][j+1:k+1])) + list(reversed(best_route[0][k+1:L+1])) + best_route[0][L+1:]
                        best_route_05[1] = distance_calc(Xdata, best_route_05)                     
                        best_route_06[0] = best_route[0][:i+1] + best_route[0][j+1:k+1] + best_route[0][i+1:j+1] + list(reversed(best_route[0][k+1:L+1])) + best_route[0][L+1:]
                        best_route_06[1] = distance_calc(Xdata, best_route_06)                       
                        best_route_07[0] = best_route[0][:i+1] + best_route[0][j+1:k+1] + list(reversed(best_route[0][i+1:j+1])) + list(reversed(best_route[0][k+1:L+1])) + best_route[0][L+1:]
                        best_route_07[1] = distance_calc(Xdata, best_route_07)                        
                        best_route_08[0] = best_route[0][:i+1] + list(reversed(best_route[0][j+1:k+1])) + best_route[0][i+1:j+1] + list(reversed(best_route[0][k+1:L+1])) + best_route[0][L+1:]
                        best_route_08[1] = distance_calc(Xdata, best_route_08)                        
                        best_route_09[0] = best_route[0][:i+1] + best_route[0][k+1:L+1] + best_route[0][i+1:j+1] + list(reversed(best_route[0][j+1:k+1])) + best_route[0][L+1:]
                        best_route_09[1] = distance_calc(Xdata, best_route_09)                        
                        best_route_10[0] = best_route[0][:i+1] + best_route[0][k+1:L+1] + list(reversed(best_route[0][i+1:j+1])) + best_route[0][j+1:k+1] + best_route[0][L+1:]
                        best_route_10[1] = distance_calc(Xdata, best_route_10)                        
                        best_route_11[0] = best_route[0][:i+1] + best_route[0][k+1:L+1] + list(reversed(best_route[0][i+1:j+1])) + list(reversed(best_route[0][j+1:k+1])) + best_route[0][L+1:]
                        best_route_11[1] = distance_calc(Xdata, best_route_11)                        
                        best_route_12[0] = best_route[0][:i+1] + list(reversed(best_route[0][k+1:L+1])) + best_route[0][i+1:j+1] + list(reversed(best_route[0][j+1:k+1])) + best_route[0][L+1:]
                        best_route_12[1] = distance_calc(Xdata, best_route_12)
                        best_route_13[0] = best_route[0][:i+1] + list(reversed(best_route[0][k+1:L+1])) + list(reversed(best_route[0][i+1:j+1])) + best_route[0][j+1:k+1] + best_route[0][L+1:]
                        best_route_13[1] = distance_calc(Xdata, best_route_13)                        
                        best_route_14[0] = best_route[0][:i+1] + list(reversed(best_route[0][k+1:L+1])) + list(reversed(best_route[0][i+1:j+1])) + list(reversed(best_route[0][j+1:k+1])) + best_route[0][L+1:]
                        best_route_14[1] = distance_calc(Xdata, best_route_14)                       
                        best_route_15[0] = best_route[0][:i+1] + list(reversed(best_route[0][i+1:j+1])) + best_route[0][k+1:L+1] + best_route[0][j+1:k+1] + best_route[0][L+1:]
                        best_route_15[1] = distance_calc(Xdata, best_route_15)
                        best_route_16[0] = best_route[0][:i+1] + list(reversed(best_route[0][i+1:j+1])) + best_route[0][k+1:L+1] + list(reversed(best_route[0][j+1:k+1])) + best_route[0][L+1:]
                        best_route_16[1] = distance_calc(Xdata, best_route_16)                        
                        best_route_17[0] = best_route[0][:i+1] + list(reversed(best_route[0][i+1:j+1])) + list(reversed(best_route[0][k+1:L+1])) + best_route[0][j+1:k+1] + best_route[0][L+1:]
                        best_route_17[1] = distance_calc(Xdata, best_route_17)
                        best_route_18[0] = best_route[0][:i+1] + best_route[0][j+1:k+1] + list(reversed(best_route[0][k+1:L+1])) + best_route[0][i+1:j+1] + best_route[0][L+1:]
                        best_route_18[1] = distance_calc(Xdata, best_route_18)                       
                        best_route_19[0] = best_route[0][:i+1] + best_route[0][j+1:k+1] + list(reversed(best_route[0][k+1:L+1])) + list(reversed(best_route[0][i+1:j+1])) + best_route[0][L+1:]
                        best_route_19[1] = distance_calc(Xdata, best_route_19)                        
                        best_route_20[0] = best_route[0][:i+1] + list(reversed(best_route[0][j+1:k+1])) + best_route[0][k+1:L+1] + best_route[0][i+1:j+1] + best_route[0][L+1:]
                        best_route_20[1] = distance_calc(Xdata, best_route_20)                        
                        best_route_21[0] = best_route[0][:i+1] + list(reversed(best_route[0][j+1:k+1])) + best_route[0][k+1:L+1] + list(reversed(best_route[0][i+1:j+1])) + best_route[0][L+1:]
                        best_route_21[1] = distance_calc(Xdata, best_route_21)                        
                        best_route_22[0] = best_route[0][:i+1] + list(reversed(best_route[0][j+1:k+1])) + list(reversed(best_route[0][k+1:L+1])) + best_route[0][i+1:j+1] + best_route[0][L+1:]
                        best_route_22[1] = distance_calc(Xdata, best_route_22)
                        best_route_23[0] = best_route[0][:i+1] + list(reversed(best_route[0][j+1:k+1])) + list(reversed(best_route[0][k+1:L+1])) + list(reversed(best_route[0][i+1:j+1])) + best_route[0][L+1:]
                        best_route_23[1] = distance_calc(Xdata, best_route_23)                      
                        best_route_24[0] = best_route[0][:i+1] + best_route[0][k+1:L+1] + best_route[0][j+1:k+1] + list(reversed(best_route[0][i+1:j+1])) + best_route[0][L+1:]
                        best_route_24[1] = distance_calc(Xdata, best_route_24)                        
                        best_route_25[0] = best_route[0][:i+1] + best_route[0][k+1:L+1] + list(reversed(best_route[0][j+1:k+1])) + best_route[0][i+1:j+1] + best_route[0][L+1:]
                        best_route_25[1] = distance_calc(Xdata, best_route_25)
                        best_route_26[0] = best_route[0][:i+1] + list(reversed(best_route[0][k+1:L+1])) + best_route[0][j+1:k+1] + best_route[0][i+1:j+1] + best_route[0][L+1:]
                        best_route_26[1] = distance_calc(Xdata, best_route_26)                        
                        best_route_27[0] = best_route[0][:i+1] + list(reversed(best_route[0][k+1:L+1])) + best_route[0][j+1:k+1] + list(reversed(best_route[0][i+1:j+1])) + best_route[0][L+1:]
                        best_route_27[1] = distance_calc(Xdata, best_route_27)
                        
                        if(best_route_01[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_01[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_01[0][n]
                                
                        elif(best_route_02[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_02[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_02[0][n] 
                                
                        elif(best_route_03[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_03[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_03[0][n] 
                                
                        elif(best_route_04[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_04[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_04[0][n]
                                
                        elif(best_route_05[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_05[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_05[0][n] 

                        elif(best_route_06[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_06[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_06[0][n] 
                                
                        elif(best_route_07[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_07[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_07[0][n] 
                                
                        elif(best_route_08[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_08[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_08[0][n]
                                
                        elif(best_route_09[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_09[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_09[0][n] 
                                
                        elif(best_route_10[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_10[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_10[0][n] 
                                
                        elif(best_route_11[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_11[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_11[0][n] 
                                
                        elif(best_route_12[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_12[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_12[0][n]
                                
                        elif(best_route_13[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_13[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_13[0][n] 

                        elif(best_route_14[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_14[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_14[0][n] 
                                
                        elif(best_route_15[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_15[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_15[0][n] 
                                
                        elif(best_route_16[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_16[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_16[0][n]
                                
                        elif(best_route_17[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_17[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_17[0][n] 

                        elif(best_route_18[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_18[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_18[0][n]
                                
                        elif(best_route_19[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_19[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_19[0][n] 
                                
                        elif(best_route_20[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_20[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_20[0][n] 
                                
                        elif(best_route_21[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_21[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_21[0][n] 
                                
                        elif(best_route_22[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_22[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_22[0][n]
                                
                        elif(best_route_23[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_23[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_23[0][n] 

                        elif(best_route_24[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_24[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_24[0][n] 
                                
                        elif(best_route_25[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_25[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_25[0][n] 
                                
                        elif(best_route_26[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_26[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_26[0][n]
                                
                        elif(best_route_27[1]  < best_route[1]):
                            best_route[1] = copy.deepcopy(best_route_27[1])
                            for n in range(0, len(best_route[0])): 
                                best_route[0][n] = best_route_27[0][n] 
                                
                    if (best_route[1] < city_list[1]):
                        city_list[1] = copy.deepcopy(best_route[1])
                        for n in range(0, len(city_list[0])): 
                            city_list[0][n] = best_route[0][n]              
                    best_route = copy.deepcopy(seed)
        count = count + 1  
        print("Iteration = ", count, "->", city_list)
    return city_list
######################## Part 1 - Usage ####################################

X = pd.read_csv('Python-MH-Local Search-4-opt-Dataset-01.txt', sep = '\t') #17 cities = 1922.33
seed = seed_function(X)
ls4opt = local_search_4_opt(X, city_tour = seed, recursive_seeding = 5)
plot_tour_distance_matrix(X, ls4opt) # Red Point = Initial city; Orange Point = Second City # The generated coordinates (2D projection) are aproximated, depending on the data, the optimum tour may present crosses.

Y = pd.read_csv('Python-MH-Local Search-4-opt-Dataset-02.txt', sep = '\t') # Berlin 52 = 7544.37
X = buid_distance_matrix(Y)
seed = seed_function(X)
ls4opt = local_search_4_opt(X, city_tour = seed, recursive_seeding = 5)
plot_tour_coordinates (Y, ls4opt) # Red Point = Initial city; Orange Point = Second City
