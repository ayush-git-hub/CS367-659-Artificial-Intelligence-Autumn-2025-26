import numpy as np

class GridWorld:
    def __init__(self, reward_step=-0.04):
        self.rows = 3
        self.cols = 4
        self.reward_step = reward_step
        self.gamma = 0.9 # Set to 0.9 to ensure convergence for positive rewards
        
        self.grid = np.zeros((self.rows, self.cols))
        self.terminal_states = {(0, 3): 1.0, (1, 3): -1.0} # (row, col) 0-indexed from top-left? 
        # Figure 1 shows (1,1) at bottom-left. Let's use (row, col) where row 0 is bottom, row 2 is top.
        # Col 0 is left, Col 3 is right.
        # 3 rows (0, 1, 2), 4 cols (0, 1, 2, 3).
        # +1 at (2, 3) (Top-Right)
        # -1 at (1, 3) (Middle-Right)
        # Wall at (1, 1)
        # Start at (0, 0)
        
        self.terminals = {
            (2, 3): 1.0,
            (1, 3): -1.0
        }
        self.walls = {(1, 1)}
        self.actions = ['U', 'D', 'L', 'R']
        
    def is_valid(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if (r, c) not in self.walls:
                return True
        return False

    def get_next_state(self, r, c, action):
        # Deterministic move
        nr, nc = r, c
        if action == 'U': nr += 1
        elif action == 'D': nr -= 1
        elif action == 'L': nc -= 1
        elif action == 'R': nc += 1
        
        if self.is_valid(nr, nc):
            return nr, nc
        else:
            return r, c # Bump into wall/boundary

    def get_transitions(self, r, c, action):
        # Returns list of (prob, next_r, next_c)
        transitions = []
        
        # Intended
        intended_r, intended_c = self.get_next_state(r, c, action)
        transitions.append((0.8, intended_r, intended_c))
        
        # Perpendiculars
        if action in ['U', 'D']:
            l_r, l_c = self.get_next_state(r, c, 'L')
            r_r, r_c = self.get_next_state(r, c, 'R')
            transitions.append((0.1, l_r, l_c))
            transitions.append((0.1, r_r, r_c))
        elif action in ['L', 'R']:
            u_r, u_c = self.get_next_state(r, c, 'U')
            d_r, d_c = self.get_next_state(r, c, 'D')
            transitions.append((0.1, u_r, u_c))
            transitions.append((0.1, d_r, d_c))
            
        return transitions

    def value_iteration(self, epsilon=1e-6, max_iter=10000):
        # Initialize value function
        V = np.zeros((self.rows, self.cols))
        
        for (r, c), val in self.terminals.items():
            V[r, c] = val
            
        iteration = 0
        while iteration < max_iter:
            delta = 0
            new_V = V.copy()
            
            for r in range(self.rows):
                for c in range(self.cols):
                    if (r, c) in self.terminals or (r, c) in self.walls:
                        continue
                    
                    # Calculate max expected utility
                    action_values = []
                    for a in self.actions:
                        val = 0
                        for prob, nr, nc in self.get_transitions(r, c, a):
                            # Reward is step cost for non-terminals
                            # If next state is terminal, we get its utility?
                            # AIMA: R(s) is immediate reward for being in s.
                            # The problem says: "immediate reward for moving to any state s... is r(s)"
                            # "reward for moving to terminal states is +1 and -1"
                            
                            # Let's interpret: R(s) is received when entering s.
                            # Bellman: V(s) = max_a sum_s' P(s'|s,a) [R(s') + gamma*V(s')]
                            # But here R depends on state we move TO.
                            
                            reward = self.reward_step
                            if (nr, nc) in self.terminals:
                                reward = self.terminals[(nr, nc)] # The problem says reward for moving TO terminal is +1/-1.
                                # Wait, if we get +1 for moving to terminal, and then terminate, V(terminal) should be 0?
                                # Or is +1 the value of the terminal state itself?
                                # "reward for moving to terminal states is +1 and -1 respectively"
                                # "immediate reward for moving to any state (s) except for the terminal states... is -0.04"
                                
                                # This implies R(s') is the reward.
                                # If s' is terminal, R(s') = +/- 1. Value of terminal state is 0 (absorbing).
                                # If s' is non-terminal, R(s') = -0.04.
                                
                                # Let's stick to standard: V(s) = max_a sum P(s'|s,a) [R(s') + gamma*V(s')]
                                # V(terminal) = 0.
                                pass
                            
                            # Actually, usually in these grid worlds:
                            # Terminals have fixed utility values and you don't move out.
                            # AIMA Ch 17: U(s) = R(s) + gamma * max ...
                            # R(s) = -0.04 for non-terminals.
                            # R(terminal) = +1/-1.
                            # And terminals are absorbing with 0 future cost? Or just the utility is fixed.
                            # Let's assume V(terminal) is fixed at +1/-1 and we don't update it.
                            # And R(s) is the "living reward" or step cost.
                            # Problem says: "immediate reward for moving to any state (s) except for the terminal states S+ is r(s)= -0.04"
                            # This phrasing is slightly ambiguous. "Moving to s".
                            # Usually it means R(s, a, s') = -0.04 if s' is not terminal.
                            # If s' is terminal, R = +1/-1.
                            
                            # Let's use: V(s) = max_a sum P(s'|s,a) * (Reward(s') + gamma * V(s'))
                            # If s' is terminal: Reward = +1/-1, V(s') = 0 (game ends).
                            # If s' is not terminal: Reward = -0.04, V(s') is current estimate.
                            
                            r_s_prime = self.reward_step
                            v_s_prime = V[nr, nc]
                            
                            if (nr, nc) in self.terminals:
                                r_s_prime = self.terminals[(nr, nc)]
                                v_s_prime = 0 # Absorbing
                            
                            val += prob * (r_s_prime + self.gamma * v_s_prime)
                            
                        action_values.append(val)
                    
                    new_V[r, c] = max(action_values)
                    delta = max(delta, abs(new_V[r, c] - V[r, c]))
            
            V = new_V
            iteration += 1
            if delta < epsilon:
                break
        
        return V

    def print_values(self, V):
        # Print in grid format (top row first)
        print(f"Values for r(s) = {self.reward_step}:")
        for r in range(self.rows - 1, -1, -1):
            row_str = ""
            for c in range(self.cols):
                if (r, c) in self.walls:
                    row_str += "  WALL  "
                elif (r, c) in self.terminals:
                    row_str += f" {self.terminals[(r,c)]:6.2f} " # Show the terminal value/reward
                else:
                    row_str += f" {V[r, c]:6.2f} "
            print(row_str)
        print("-" * 40)

if __name__ == "__main__":
    rewards = [-0.04, -2, 0.1, 0.02, 1]
    for r in rewards:
        gw = GridWorld(reward_step=r)
        V = gw.value_iteration()
        gw.print_values(V)
