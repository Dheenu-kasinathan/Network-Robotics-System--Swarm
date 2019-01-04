import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mp
import IPython

np.set_printoptions(linewidth=100)
p = np.reshape(np.array([10,1,1,0,2,2,3,2,4,0]),(10,1))
pfinal = np.reshape(np.array([0,2,1,1,0,0,-1,1,0,1]),(10,1))
E = np.array([[0,3],[0,4],[1,2],[1,4],[2,4],[3,4],[3,2]])


def gG(p):
    v = p.reshape(5,2)
    gG = np.zeros(len(E))
    for i in range(len(E)):
        gG[i]=np.square(LA.norm(v[E[i][0]]-v[E[i][1]]))
    gG = gG.reshape(len(gG),1)
    
    return gG

def RG(p):
    v = p.reshape(5,2)
    RG = np.zeros((7,10))
    for i in range(len(E)):
        e = E[i]
        RG[i][2*E[i][0]] = 2*(v[E[i][0]][0]-v[E[i][1]][0])
        RG[i][2*E[i][0]+1] = 2*(v[E[i][0]][1]-v[E[i][1]][1])
        RG[i][2*E[i][1]] = -2*(v[E[i][0]][0]-v[E[i][1]][0])
        RG[i][2*E[i][1]+1] = -2*(v[E[i][0]][1]-v[E[i][1]][1])

    return RG

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

gd = gG(pfinal)
print("Vector of desired constraints")
print(gd)

P = p
T = 0

while T<=10 :
    u = np.matmul(np.transpose(RG(p)),(gd-gG(p)))
    pnew = p + u*0.001
    P = np.hstack((P,pnew))
    p = pnew
    T = T + 0.001

obstacles = []
plotx = P[:,0::100]

make_animation_obstacles(plotx, E, obstacles, inter=100, display=True, xl=(-1, 12), yl=(-1, 12)).save("video1.mp4")