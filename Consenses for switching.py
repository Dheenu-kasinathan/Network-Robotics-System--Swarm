import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

D = 2 #int(input("Enter the number of switching graphts: "))
switch_time = 0.1 #int(input("Enter the switch time: "))
T = 30 #int(input("Enter the time: "))
N = 7 #int(input("Enter the number of vertices: "))
a = 0

#X_0 = np.matrix(input("Enter the initial conditions: "))
X_0 = [1, 4, 3, 2, 6, 7, 8, 0]   # 1;4;3;2;6;7;8;0
dt = 0.001

NoofEdges = [[] for i in range(D)]
E = [[] for i in range(D)]
Laplacian = [[] for i in range(D)]
#x = [[] for i in range(D)]

#for i in range(D):
#    NoofEdges[i] = int(input("Enter the number of Edges: "))
#    E[i] = np.matrix(input("Enter the head and tail of the edges: ")) 

NoofEdges[0] = 7
NoofEdges[1] = 7
#NoofEdges[2] = 12

#E[0]=np.matrix([[1,3],[3,1],[2,3],[3,2],[3,4],[4,3],[2,5],[5,2],[3,6],[6,3],[4,7],[7,4],[5,6],[6,5]])    
#E[1]=np.matrix([[1,3],[3,1],[2,3],[3,2],[3,4],[4,3],[2,5],[5,2],[4,7],[7,4],[5,6],[6,5]])                
#E[2]=np.matrix([[1,3],[3,1],[2,3],[3,2],[3,4],[4,3],[3,6],[6,3],[4,7],[7,4],[5,6],[6,5]])               
                                                    # 1,3; 3,1; 2,3; 3,2; 3,4; 4,3; 2,5; 5,2; 3,6; 6,3; 4,7; 7,4; 5,6; 6,5
                                                    # 1,3; 3,1; 2,3; 3,2; 3,4; 4,3; 2,5; 5,2; 4,7; 7,4; 5,6; 6,5
                                                    # 1,3; 3,1; 2,3; 3,2; 3,4; 4,3; 3,6; 6,3; 4,7; 7,4; 5,6; 6,5
                                                    
E[0]=np.matrix([[1,3],[3,2],[3,4],[2,5],[3,6],[7,4],[5,6]])     # 1,3; 3,2; 3,4; 2,5; 3,6; 7,4; 5,6
E[1]=np.matrix([[1,3],[3,2],[3,4],[2,5],[6,3],[4,7],[6,5]])   # 1,3; 3,2; 3,4; 2,5; 6,3; 4,7; 6,5

#E[0]=np.matrix([[2,5],[3,6],[7,4],[5,6]])           # 2,5; 3,6; 7,4; 5,6
#E[1]=np.matrix([[3,2],[5,5],[4,7]])                 # 3,2; 6,5; 4,7
#E[2]=np.matrix([[1,3],[3,4],[6,3]])                 # 1,3; 3,4; 6,3
            
TypeofGraph = 'd' #input("Enter the type of Graph (directed/undirected or d/u): ")
if TypeofGraph in ('Directed', 'D', 'd'):
    Boolean = True
elif TypeofGraph in ('Unidirected', 'U', 'u'):
    Boolean = False
else:
    print('check verify your response')
    
#-------------------------------------------------------------------------------------------------------------------------------

def get_laplacian(E, N, Boolean):
  
    zeros = np.zeros((N,N))
    Laplacian = np.zeros((N,N))
    Degree = np.zeros((N,N))
    
    if Boolean == True:    #directed graph
        
        for i in range(len(E)):
            u=E[i,0]
            v=E[i,1]
            #u = E.item(i,0)
            #v = E.item(i,1)           
            zeros[u-1,v-1] = 1            
        AdjacencyMatrix=np.transpose(zeros)
       
        for i in range(N):
            for j in range(N):
                Degree[i][i] = Degree[i][i] + AdjacencyMatrix[i][j]
    
    else:                
        for i in range(len(E)):
            u = E.item(i,0)
            v = E.item(i,1)           
            zeros[u-1,v-1] = 1
            zeros[v-1,u-1] = 1            
        AdjacencyMatrix=zeros
        
        for i in range(N):
            for j in range(N):
                Degree[i][i] = Degree[i][i] + AdjacencyMatrix[i][j]
       
    Laplacian = Degree - AdjacencyMatrix
    
    return Laplacian

#-------------------------------------------------------------------------------------------------------------------------------

for i in range(D):
    Laplacian[i] = get_laplacian(E[i], N, Boolean)
    
#print(Laplacian)

#------------------------------------------------------------------------------------------------------------------------------

def simulate_consensus(X_0, T, Laplacian, switch_time, dt, N, D):
    
    n =int(T/dt)
    X = np.zeros((N,n))
    Laplacianchange = np.zeros((N,N))

    t= [i*dt for i in range(n)]
    
    temp_1 = [0 for i in range(n)]
   
    repeat = function_for_repeat(D, T, t, switch_time, n, temp_1)
    #print(repeat)
    
    for i in range(N):
        X[i][0] = X_0[i]

    for i in range(1,n):
        
        Lap_list = changelaplacian(Laplacian, i, Laplacianchange, repeat)
            
        for k in range(N):
            X[k][i] = X[k][i-1] + dt*(functionfornextstates(i,k,X,Lap_list)) 
                        
    return X

def function_for_repeat(D, T, t, switch_time, n, temp_1):
   
    
    temp_2 = [0 for i in range(n)]
    
    for i in range(n):
        t[i]=round(t[i]*10000,0)
    switch_time = switch_time * 10000
    
    for i in range(n):
        temp_1[i] = t[i] % switch_time
    
    for i in range(1,n):
        if temp_1[i] == 0:
            temp_2[i]=temp_2[i-1]+1
            if temp_2[i] < D:
                temp_2[i] = temp_2[i]
            else:
                temp_2[i] = 0
        else:
            temp_2[i] = temp_2[i-1]

    return temp_2

def changelaplacian(Laplacian, i, Laplacianchange, repeat):
    a =repeat[i]
    Laplacianchange = Laplacian[a]
    return Laplacianchange

def functionfornextstates(i,k,X,Lap_list):

    temp = 0
    for l in range(N):
        if Lap_list[k][l] == -1:
            a=l
            b=k
            temp += X[a][i-1] - X[b][i-1]
    return temp

#------------------------------------------------------------------------------------------------------------------------------

x = simulate_consensus(X_0, T, Laplacian, switch_time, dt, N, D)
       
plt.xlabel('Time')
plt.ylabel('initial conditions')
for i in range(N):
    plt.plot(x[i], label = i+1)

legend = plt.legend(loc = 'upper right', shadow = True)
legend.get_frame().set_facecolor('#ffffff')

plt.show()
