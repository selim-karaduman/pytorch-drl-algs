import abc

class BaseModel:
    """
    Base model; anything that uses MCTS should overwrite these methods
    """

    """
    returns next_state, reward, done
    """
    def forward(self, action):
        pass

    """
    returns list of available actions
    """
    def get_available_actions(self):
        pass

    """
    constructs back the state of the model from the stack
    """
    def load_states(self):
        pass

    """
    pushes the current model information to the stack
    """
    def save_states(self):
        pass

    """
    given the initial state; what is the score: this is useful for 
        board games where the score is dependent on the player
    """
    def get_score(self, init_state):
        pass

    """
    returns bool
    """
    def is_terminal(self):
        pass


class MuZeroModel(BaseModel):

    def __init__(self, dynamics_model):
        self.dynamics_model = dynamics_model
    
    def forward(self, action):
        TODO

    def get_available_actions(self):
        TODO
        
    def load_states(self):
        pass

    def save_states(self):
        pass

    def get_score(self, init_state):
        return 0

    def is_terminal(self):
        return False