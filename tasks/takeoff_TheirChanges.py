import numpy as np
from physics_sim import PhysicsSim

class Takeoff():
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
        print("########2 Task to Takeoff")
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.runtime = runtime
        self.position_noise = position_noise
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = 0.0

        #encorage acceleration in the +z-direction
        #reward += self.sim.linear_accel[2]
        #reward +=  0.5 * self.sim.linear_accel[2] * self.sim.dt**2

        #encorage velocity in the +z-direction
        if  self.sim.v[2] > 0:
            reward = 10*self.sim.v[2]
        else:
            reward = -10

        
        
        
        
        



        #encorage smaller position errors and limit the demenonator to not be too much close to zero
        #posDiff = -abs(self.sim.pose[ 2] - self.target_pos[ 2]) #**2 
        #if posDiff > 0.1: 
        #    reward += 10/posDiff 
        #else :
        #    reward += 100


        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)

            #sparse reward at the end if target is reached 
            if(self.sim.pose[2] >= self.target_pos[2]):
                print("\n sucessfull takeoff to target target  !!!!!!!!!")
                reward += 10
                done = True

        #if np.isclose(self.target_pos[2], self.sim.pose[2], 1):
        #    reward += 100
        #    done = True



        #if(self.sim.pose[2] >= self.target_pos[2] and self.sim.time >= self.runtime):
        #if np.allclose(self.sim.pose[:-3],self.target_pos, rtol=1):
        #    reward +=100
        #    done = True

        next_state = np.concatenate(pose_all)
        return next_state, reward, done


    def reset(self):
        """Reset the sim with noise to start a new episode."""
        self.sim.reset()

        #::os:: add random start position as suggested by some papers to make agent more generalized
        if self.position_noise is not None or self.ang_noise is not None:
            rand_pose = np.copy(self.sim.init_pose)
            if self.position_noise is not None and self.position_noise > 0:
                rand_pose[:3] += np.random.normal(0.0, self.position_noise, 3)
                #print("start rand_pose =",rand_pose) 

            self.sim.pose = np.copy(rand_pose)

        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state




