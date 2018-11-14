import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

N = int(input("Enter the number of vertices: "))

NoofEdges = int(input("Enter the number of Edges: "))

E = np.matrix(input("Enter the head and tail of the edges: "))  # 3,1;3,2;4,3;5,3;6,5;7,3;7,5;7,6;8,5;8,7;9,7

TypeofGraph = input("Enter the type of Graph (directed/undirected or d/u): ")

print("Enter the Initial States: ")
#X_0 = [input() for i in range(N)]   #X_0 = [20, 10, 15, 12, 30, 12, 15, 16, 25]
X_0 = [1, 4, 3, 2, 6, 7, 8, 0]

T = int(input("Enter the integration Time: "))
dt =0.001

if TypeofGraph in ('Directed', 'D', 'd'):
    Boolean = True
elif TypeofGraph in ('Unidirected', 'U', 'u'):
    Boolean = False
else:
    print('check verify your response')

zeros = np.zeros((N,N))
Laplacian = np.zeros((N,N))
Degree = np.zeros((N,N))
    
def get_laplacian(E, N, Boolean):
    
    if Boolean == True:                                                 #directed graph
        
        for i in range(len(E)):
            u = E.item(i,0)
            v = E.item(i,1)
           
            zeros[u-1,v-1] = 1
            
        AdjacencyMatrix=np.transpose(zeros)
        #print("Adjacency Matrix: ")
        #print(AdjacencyMatrix)
        
        for i in range(N):
            for j in range(N):
                Degree[i][i] = Degree[i][i] + AdjacencyMatrix[i][j]
        #print("Degree Matrix: ")
        #print(Degree)
       
    else:                                                               #undirected graph
        for i in range(len(E)):
            u = E.item(i,0)
            v = E.item(i,1)
           
            zeros[u-1,v-1] = 1
            zeros[v-1,u-1] = 1
            
        AdjacencyMatrix=zeros
        #print("Adjacency Matrix: ")
        #print(AdjacencyMatrix)
        
        for i in range(N):
            for j in range(N):
                Degree[i][i] = Degree[i][i] + AdjacencyMatrix[i][j]
        #print("Degree Matrix: ")
        #print(Degree)
       
    Laplacian = Degree - AdjacencyMatrix
    
    return Laplacian
    
#exercise 5

def simulate_consensus(X_0, T, Laplacian, dt, N):

    n =int(T/dt)
    X = np.zeros((N,n))

    t = [i*0.001 for i in range(n)]
    
    for i in range(N):
        X[i][0] = X_0[i]
    
    for i in range(1,n):
        for k in range(N):
            X[k][i] = X[k][i-1] + dt*(functionfornextstates(i,k,X,Laplacian)) #xdot = -L*x,  xdot = functionofneighbouringstates,  (x - (x-dt))/dt =function, x = (x-dt) + dt*(function)
                        
    return X,t

def functionfornextstates(i,k,X,Laplacian):

    temp = 0
    for l in range(N):
        if Laplacian[k][l] == -1:
            a=l
            b=k
            temp += X[a][i-1] - X[b][i-1]
    return temp


Laplacian = get_laplacian(E, N, Boolean)
print("Laplacian Matrix: ")
print (Laplacian)

#eigen value
eigen,vector = np.linalg.eig(Laplacian)
#print("Eigen values are: ")
#print(eigen)

sort = np.sort(eigen)
#print("Sorted Eigen Values are: ")
#print(sort)

#print("smallest eigen value: ", sort[0]) 
#print("2 nd smallest eigen value: ", sort[1])
#print("Largest eigen value: ", sort[-1])
#print("2nd Largest eigen value: ", sort[-2])

x,t = simulate_consensus(X_0, T, Laplacian, dt, N)

plt.xlabel('Time')
plt.ylabel('Temp')
for i in range(N):
    plt.plot(x[i], label = i+1)

legend = plt.legend(loc = 'upper right', shadow = True)
legend.get_frame().set_facecolor('#ffffff')

plt.show()

       


    

