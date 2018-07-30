import numpy as np
from physics_sim import PhysicsSim

class Hover():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None,position_noise=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        print(":::::::::: Task to hover  :::::::::: ")
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        self.position_noise = position_noise
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()   #from harsh
        reward = 0.0


        # Reward positions close to target along z-axis
        posDiff = abs(self.sim.pose[2] - self.target_pos[2]) #**2 #np.tanh(0.01*abs(self.sim.pose[ 2] - self.target_pos[ 2]))
        if posDiff > 0.1:
            reward += 10/posDiff
        else :
            reward += 100


        #reward +=  1/0.5 * self.sim.linear_accel[2] * self.sim.dt**2

        # Reward positive velocity along z-axis
        #if self.sim.v[2]>0.001 and posDiff < 1:
        #    reward += 1/(posDiff*self.sim.v[2])
        #else:
        #    reward +=100
        

        #reward += max(posReward, 100) #/ 2.0 

        # A lower sensativity towards drifting in the xy-plane
        #reward -= (abs(self.sim.pose[:2] - self.target_pos[:2])).sum() / 4.0

        # Negative reward for angular velocities
        #reward -= (abs(self.sim.angular_v[:3])).sum()
        #print("########### reward=",reward)





        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities

            reward += self.get_reward() 
            pose_all.append(self.sim.pose)


        def is_equal(x, y, delta=0.0):
            return abs(x-y) <= delta

        #if is_equal(self.target_pos[2], self.sim.pose[2], delta=3):
        #    reward += 10.0  # bonus reward
            # done = True 		#::os:: note here we are not done yet, keep hovering until TIME run out

        #if is_equal(self.target_pos[2], self.sim.pose[2], delta=2):
        #    reward += 20.0  # bonus reward
            # done = True 		#::os:: note here we are not done yet, keep hovering until TIME run out

        if is_equal(self.target_pos[2], self.sim.pose[2], delta=1):
            reward += 50.0  # bonus reward
            # done = True 		#::os:: note here we are not done yet, keep hovering until TIME run out
        #print("runtime=",self.sim.runtime)
        if self.sim.time > self.sim.runtime and  is_equal(self.target_pos[2], self.sim.pose[2], delta=1):
            #bonus reward for staying up in air
            print("########## good finish still on the air")
            reward += 1000.0
            done = True		#actually already set to true from inside physSim







        next_state = np.concatenate(pose_all)
        return next_state, reward, done

#    def reset(self):
#        """Reset the sim to start a new episode."""
#        self.sim.reset()
#        state = np.concatenate([self.sim.pose] * self.action_repeat) 
#        return state

##################
    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()

        #::os:: add random start position as suggested by some papers to make agent more generalized
        if self.position_noise is not None:
            rand_pose = np.copy(self.sim.init_pose)
            if self.position_noise is not None and self.position_noise > 0:
                rand_pose[:3] += np.random.normal(0.0, self.position_noise, 3)
                #print("start rand_pose =",rand_pose) 

            self.sim.pose = np.copy(rand_pose)
            self.target_pos = np.copy(rand_pose)


        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state


