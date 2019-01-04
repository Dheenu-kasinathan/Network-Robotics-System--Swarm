import numpy as np
import pybullet as p
import itertools

class Robot():
    """ 
    The class is the interface to a single robot
    """
    K_F = 2
    D_F = 4
   
    def __init__(self, init_pos, robot_id, dt):
        self.id = robot_id
        self.dt = dt
        self.pybullet_id = p.loadSDF("../models/robot.sdf")[0]
        self.joint_ids = list(range(p.getNumJoints(self.pybullet_id)))
        self.initial_position = init_pos
        self.reset()

        # No friction between bbody and surface.
        p.changeDynamics(self.pybullet_id, -1, lateralFriction=5., rollingFriction=0.)

        # Friction between joint links and surface.
        for i in range(p.getNumJoints(self.pybullet_id)):
            p.changeDynamics(self.pybullet_id, i, lateralFriction=5., rollingFriction=0.)
            
        self.messages_received = []
        self.messages_to_send = []
        self.neighbors = []
        

    def reset(self):
        """
        Moves the robot back to its initial position 
        """
        p.resetBasePositionAndOrientation(self.pybullet_id, self.initial_position, (0., 0., 0., 1.))
            
    def set_wheel_velocity(self, vel):
        """ 
        Sets the wheel velocity,expects an array containing two numbers (left and right wheel vel) 
        """
        assert len(vel) == 2, "Expect velocity to be array of size two"
        p.setJointMotorControlArray(self.pybullet_id, self.joint_ids, p.VELOCITY_CONTROL,
            targetVelocities=vel)

    def get_pos_and_orientation(self):
        """
        Returns the position and orientation (as Yaw angle) of the robot.
        """
        pos, rot = p.getBasePositionAndOrientation(self.pybullet_id)
        euler = p.getEulerFromQuaternion(rot)
        return np.array(pos), euler[2]
    
    def get_messages(self):
        """
        returns a list of received messages, each element of the list is a tuple (a,b)
        where a= id of the sending robot and b= message (can be any object, list, etc chosen by user)
        Note that the message will only be received if the robot is a neighbor (i.e. is close enough)
        """
        return self.messages_received
        
    def send_message(self, robot_id, message):
        """
        sends a message to robot with id number robot_id, the message can be any object, list, etc
        """
        self.messages_to_send.append([robot_id, message])
        
    def get_neighbors(self):
        """
        returns a list of neighbors (i.e. robots within 2m distance) to which messages can be sent
        """
        return self.neighbors

    #def getLaplacian(self,E,n_vertex):
     #   L = np.zeros([n_vertex,n_vertex]) #our Laplacian matrix
      #  Delta = np.zeros([n_vertex,n_vertex]) #this is the degree matrix
      #  A = np.zeros([n_vertex,n_vertex]) #this is the adjacency matrix
       # for e in E: #for each edge in E
            #add degrees
        #    Delta[e[1],e[1]] +=1
            #add the input in the adjacency matrix
         #   A[e[1],e[0]] = 1
            #symmetric connection as we have undirected graphs
          #  Delta[e[0],e[0]] +=1
           # A[e[0],e[1]] = 1
       # L = Delta - A
        #return L

    # get incidence matrix for directed graph E (list of edges)
    #def getIncidenceMatrix(self,E,n_vertex):
     #   n_e = len(E)
      #  D = np.zeros([n_vertex,n_e])
       # for e in range(n_e):
            #add the directed connection
        #    D[E[e][0],e] = -1
         #   D[E[e][1],e] = 1
        #return D

        
    
    
    
    def compute_controller(self, time):
                               
        """ 
        function that will be called each control cycle which implements the control law
        TO BE MODIFIED
        
        we expect this function to read sensors (built-in functions from the class)
        and at the end to call set_wheel_velocity to set the appropriate velocity of the robots
        """

        #E = np.array([[0,1],[0,2],[0,3],[1,3],[2,3],[2,4],[2,5],[3,5],[4,5]])
        #n_vertex = 6

        #L = self.getLaplacian(E,n_vertex)
        #print(&self)
        #print("\n")
        #D = self.getIncidenceMatrix(E,n_vertex)

        #desired = np.array([[0.5,-1],[0.5,1],[1.5,-1],[1.5,1],[2.5,-1],[2.5,1]])
        #desired_x = np.array([[0.5],[0.5],[1.5],[1.5],[2.5],[2.5]])
        #desired_y = np.array([[-1],[1],[-1],[1],[-1],[1]])

        #Z_x_des = np.matmul(np.transpose(D),desired_x)
        #Z_y_des = np.matmul(np.transpose(D),desired_y)   
 
        #print("\n", time)
        # here we implement an example for a consensus algorithm
        neig = self.get_neighbors()
        messages = self.get_messages()
        pos, rot = self.get_pos_and_orientation()

        #x=[0]*6
        #y=[0]*6
        #try:
            #for i in range(6):
               # for m in messages:
              #      if i == m[0]:
             #           x[i] = m[1][0]
            #        elif i == m[0]:
           #             x[i]= 0
          #          elif i == id:
         #               x[i] = pos[0]
        #except IndexError:
         #   pass

        #print("X: ",x)
        #print(x)
        #try:        
            #for i in range(6):
              #  if i in messages:
             #       y[i] = m[1][1]
            #    elif i != id:
           #         y[i]= 0
          #      else:
         #           y[id] = pos[0]
        #except IndexError:
        #        pass
        #print(y)

        #########################################################################################
        
        #send message of positions to all neighbors indicating our position
        for n in neig:
            self.send_message(n, pos)
        
        # check if we received the position of our neighbors and compute desired change in position
        # as a function of the neighbors (message is composed of [neighbors id, position])


        #########################################################################################

        dx = 0.
        dy = 0.

        if time < 10:
            self.target = [[0.5,-1],[0.5,1],[1.5,-1],[1.5,1],[2.5,-1],[2.5,1]]
        elif time < 20 :
            self.target = [[0.5,-0.5],[2.5,0.5],[1.5,-0.5],[2.5,1.5],[2.5,-0.5],[2.5,2.5]]
        elif time < 30:
            self.target = [[2.5,0.5],[2.3,3.5],[2.5,1.5],[2.5,4.5],[2.5,2.5],[2.5,5.5]]
        elif time < 40:
            self.target = [[2.5,2.5],[2.5,5.5],[2.5,3.5],[1.5,5.5],[2.5,4.5],[0.5,5.5]]
        elif time < 50:
            self.target = [[2.5,4.5],[0.5,5.5],[2.5,5.5],[-0.5,5.5],[1.5,5.5],[-1.5,5.5]]
        elif time < 60:
            self.target = [[1.5,5.5],[-1.5,5.5],[0.5,5.5],[-2.5,5.5],[-0.5,5.5],[-3.5,5.5]]
        elif time < 70:
            self.target = [[-0.5,5.5],[-3.5,5.5],[-1.5,5.5],[-4.5,5.5],[-2.5,5.5],[-5.5,5.5]]
        elif time < 80:
            self.target = [[-2.5,5.5],[-5.5,5.5],[-3.5,5.5],[-6.5,5.5],[-4.5,5.5],[-7.5,5.5]] #straight line
        elif time < 90:
            self.target = [[-3.5,5.5],[-3.5,7.5],[-4.5,5.5],[-4.5,7.5],[-5.5,5.5],[-5.5,7.5]] #square
        elif time < 100:
            self.target = [[-3.5,10],[-3.5,11],[-4.5,10],[-4.5,11],[-5.5,10],[-5.5,11]] #move square
        elif time < 120:
            self.target = [[-3.5,9.5],[-4,10.5],[-4.5,9.5],[-4.5,11.5],[-5.5,9.5],[-5,10.5]] #form triange
    
        
        if messages:
            for m in messages:                
                
                dx += (self.K_F*(m[1][0] - pos[0] - self.target[m[0]][0] + self.target[self.id][0])) + (self.D_F*np.minimum((self.target[self.id][0]-pos[0]),1))
                dy += (self.K_F*(m[1][1] - pos[1] - self.target[m[0]][1] + self.target[self.id][1])) + (self.D_F*np.minimum((self.target[self.id][1]-pos[1]),1))

            # integrate
            #des_pos_x = pos[0] + self.dt * dx
            #des_pos_y = pos[1] + self.dt * dy
        
            #compute velocity change for the wheels
            vel_norm = np.linalg.norm([dx, dy]) #norm of desired velocity
            if vel_norm < 0.01:
                vel_norm = 0.01
            des_theta = np.arctan2(dy/vel_norm, dx/vel_norm)
            right_wheel = np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
            left_wheel = -np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
            self.set_wheel_velocity([left_wheel, right_wheel])
        

            #########################################################################################
    
       
