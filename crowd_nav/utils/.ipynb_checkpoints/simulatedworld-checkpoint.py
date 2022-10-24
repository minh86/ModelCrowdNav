import sys
sys.path.append('../../')
from crowd_sim.envs.utils.info import *

class SimulatedWorld(object):
    def __init__(self, sim_model):
        self.sim_model = sim_model

    def step(self, current_s, last_s, action):
        '''
        current_s: current state
        last_s: last_state        
        '''
        input_tensor = self.prepare_input(current_s, last_s)
        output_tensor = self.sim_model(input_tensor)
        ob = self.make_ob (output_tensor)
        # collision detection
        
        # check if reaching the goal
        
        return ob, reward, done, info
    
    def prepare_input(current_s, last_s):
        
    def make_ob (output_tensor):
        
    def make_state():