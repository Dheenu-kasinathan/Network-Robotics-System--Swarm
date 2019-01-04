import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mp
import IPython

np.set_printoptions(linewidth=100)
p = np.reshape(np.array([0,0,-0.2,-0.2,0.2,0,0,0.1]),(8,1))
pf = np.reshape(np.array([0.5,0.5,-0.5,0.5,-0.5,-0.5,0.5,-0.5]),(8,1))
E = np.array([[0,1],[0,2],[1,2],[2,3],[3,0]])
n = int(len(p)/2)

def getLaplacian(E,n_vertex):
    L = np.zeros([n_vertex,n_vertex]) #our Laplacian matrix
    Delta = np.zeros([n_vertex,n_vertex]) #this is the degree matrix
    A = np.zeros([n_vertex,n_vertex]) #this is the adjacency matrix
    for e in E: #for each edge in E
        #add degrees
        Delta[e[1],e[1]] +=1
        #add the input in the adjacency matrix
        A[e[1],e[0]] = 1
        #symmetric connection as we have undirected graphs
        Delta[e[0],e[0]] +=1
        A[e[0],e[1]] = 1
    L = Delta - A
    return L

def getIncidenceMatrix(E,n_vertex):
    n_e = len(E)
    D = np.zeros([n_vertex,n_e])
    for e in range(n_e):
        #add the directed connection
        D[E[e][0],e] = -1
        D[E[e][1],e] = 1
    return D

def make_animation_obstacles(plotx,E,obstacles,xl=(-2,2),yl=(-2,2),inter=100, display=False):
    """
    Function to create a movie of the agents moving with obstacles
    plotx: a matrix of states ordered as (x1, y1, x2, y2, ..., xn, yn) in the rows and time in columns
    E: list of edges (each edge is a pair of vertexes)
    obstacles: a list of coordinates for each obstacle [[o1x, o1y],[o2x,o2y],...]
    xl, yl: display boundaries of the graph
    inter: interval between each point in ms
    display: if True displays the anumation as a movie, if False only returns the animation
    
    in order to keep computation quick, use in plotx points spaced by at least 100ms (and use inter=100) in this case
    """
    fig = mp.figure.Figure()
    mp.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=xl, ylim=yl)
    ax.grid()

    list_of_lines = []
    for i in E: #add as many lines as there are edges
        line, = ax.plot([], [], 'o-', lw=2)
        list_of_lines.append(line)
    for obs in obstacles: #as many rounds as there are obstacles
        line, = ax.plot([obs[0]],[obs[1]],'ko-', lw=15)
        list_of_lines.append(line)

    def animate(i):
        for e in range(len(E)):
            vx1 = plotx[2*E[e][0],i]
            vy1 = plotx[2*E[e][0]+1,i]
            vx2 = plotx[2*E[e][1],i]
            vy2 = plotx[2*E[e][1]+1,i]
            list_of_lines[e].set_data([vx1,vx2],[vy1,vy2])
        return list_of_lines
    
    def init():
        return animate(0)


    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(plotx[0,:])),
        interval=inter, blit=True, init_func=init)
    plt.close(fig)
    plt.close(ani._fig)
    if(display==True):
        IPython.display.display_html(IPython.core.display.HTML(ani.to_html5_video()))
    return ani

KF = 10
KT = 5
KO = 10
DF = 2 * np.sqrt(KF)
DT = 2 * np.sqrt(KT)
dmax = 1

T = 0
L = getLaplacian(E,n)
D = getIncidenceMatrix(E,n)
z = np.matmul(np.transpose(D),np.reshape(pf,(4,2)))
v = np.reshape(np.zeros(len(p)),(len(p),1))
Ff = np.zeros((4,2))
Ft = np.zeros((8,1))
goal = np.reshape(np.array([10,10,10,10,10,10,10,10]),(8,1))
P = p

obs = np.array([[4,0],[4,4.7],[4,8],[6,2],[7,6],[6,10],[8,4],[8,12],[8.1,5.6]])

FO = []
while T<=10:
    
    Ff = KF * (np.matmul(D,z)-np.matmul(L,p.reshape(4,2))) - DF*np.matmul(L,v.reshape(4,2))
    Ff = Ff.reshape(8,1)
    
    xy = p.reshape(4,2)
    Ft = KT * np.reshape(np.minimum(goal-p,1),(8,1)) - DT * v 
    Fo = np.zeros((len(obs),1))
    for i in range (len(xy)):
        Fo = np.zeros((2,1))
        for j in range(len(obs)):
            do = LA.norm(xy[i] - obs[j] )
            if do < dmax:
                Fo[0] = Fo[0] - KO * ((do-dmax)/(do**3)) * (xy[i][0]-obs[j][0])
                Fo[1] = Fo[1] - KO * ((do-dmax)/(do**3)) * (xy[i][1]-obs[j][1])
            else:
                Fo[0] = Fo[0] 
                Fo[1] = Fo[1]                    
                    
        FO = np.append(FO,Fo)
    
    FO = FO.reshape(8,1)
    pnew = p + 0.001*v
    P = np.hstack((P,pnew))
    p = pnew
    
    vnew = v + 0.001*(Ff + Ft + FO)
    v = vnew
    T = T + 0.001
    FO = []
    
plotx = P[:,0::100]

make_animation_obstacles(plotx, E, [], inter=100, display=True, xl=(-1, 12), yl=(-1, 12)).save("video2.mp4")
make_animation_obstacles(plotx, E, obs, inter=100, display=True, xl=(-1, 12), yl=(-1, 12)).save("video3.mp4")
    

    
    
    
    
    
    
    

    