
class Experience:

    def __init__(self, state, action, reward, 
                 next_state, done, index=-1, priority=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.index = index
        self.priority = priority
        self.ind = -1

    def __iter__(self):
        self.ind = -1
        return self

    def __next__(self):
        self.ind += 1
        if self.ind == 0:
            return self.state
        if self.ind == 1:
            return self.action
        if self.ind == 2:
            return self.reward
        if self.ind == 3:
            return self.next_state
        if self.ind == 4:
            return self.done
        if self.ind == 5 and (self.priority is not None):
            return self.index
        if self.ind == 6 and (self.priority is not None):
            return self.priority
        raise StopIteration
