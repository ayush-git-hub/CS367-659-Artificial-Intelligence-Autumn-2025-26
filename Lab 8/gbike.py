import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import time

class GbikeRental:
    def __init__(self, modified=False):
        self.max_bikes = 20
        self.max_move = 5
        self.rental_reward = 10
        self.move_cost = 2
        self.discount = 0.9
        self.modified = modified
        
        self.lambda_req = [3, 4]
        self.lambda_ret = [3, 2]
        
        self.poisson_cache = {}
        self.actions = np.arange(-self.max_move, self.max_move + 1)
        self.states = [(i, j) for i in range(self.max_bikes + 1) for j in range(self.max_bikes + 1)]
        self.num_states = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        
        # Precompute P and R
        # P[s_idx, a_idx, next_s_idx]
        # R[s_idx, a_idx]
        # This might be too large: 441 * 11 * 441 ~ 2M floats. 16MB. Fine.
        print("Precomputing transitions and rewards...")
        self.P = np.zeros((self.num_states, len(self.actions), self.num_states))
        self.R = np.zeros((self.num_states, len(self.actions)))
        
        self.precompute_model()
        
        self.V = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int) # Indices of actions
        # Initialize policy to 0 move (action index 5 if -5..5)
        self.policy[:] = 5 

    def poisson(self, n, lam):
        key = (n, lam)
        if key not in self.poisson_cache:
            self.poisson_cache[key] = (lam**n / math.factorial(n)) * math.exp(-lam)
        return self.poisson_cache[key]

    def precompute_model(self):
        upper_bound = 11
        
        # Precompute poisson probs
        p_req1 = [self.poisson(n, self.lambda_req[0]) for n in range(upper_bound)]
        p_req2 = [self.poisson(n, self.lambda_req[1]) for n in range(upper_bound)]
        p_ret1 = [self.poisson(n, self.lambda_ret[0]) for n in range(upper_bound)]
        p_ret2 = [self.poisson(n, self.lambda_ret[1]) for n in range(upper_bound)]
        
        for s_idx, (n1, n2) in enumerate(self.states):
            for a_idx, action in enumerate(self.actions):
                # Apply move
                # Check validity
                # "If you are out of bikes... business lost" -> handled in rental
                # "Move max 5" -> handled by action set
                # "No more than 20 bikes" -> handled by cap?
                # Usually: move, then cap? Or move is invalid if exceeds?
                # Let's assume move is valid if we have enough bikes to move.
                
                # Check if move is valid (have enough bikes)
                if action > 0 and action > n1: # Move 1->2
                    # Invalid move, make it impossible or high penalty?
                    # Or just don't allow this action in policy improvement.
                    # Here we can leave P=0, R=-inf
                    self.R[s_idx, a_idx] = -float('inf')
                    continue
                if action < 0 and abs(action) > n2: # Move 2->1
                    self.R[s_idx, a_idx] = -float('inf')
                    continue
                
                # Apply move
                curr_n1 = min(self.max_bikes, n1 - action)
                curr_n2 = min(self.max_bikes, n2 + action)
                
                # Cost
                cost = 0
                if self.modified:
                    if action > 0:
                        cost += (action - 1) * self.move_cost if action > 1 else 0
                    else:
                        cost += abs(action) * self.move_cost
                    if curr_n1 > 10: cost += 4
                    if curr_n2 > 10: cost += 4
                else:
                    cost += abs(action) * self.move_cost
                
                expected_reward = -cost
                
                # Iterate outcomes
                # We can compute transition probs for (n1, n2) -> (next_n1, next_n2)
                # Independent for loc 1 and loc 2
                
                # Loc 1 transitions
                # P(next_n1 | curr_n1)
                probs_1 = np.zeros(self.max_bikes + 1)
                rewards_1 = 0
                
                for req in range(upper_bound):
                    for ret in range(upper_bound):
                        prob = p_req1[req] * p_ret1[ret]
                        valid_rentals = min(curr_n1, req)
                        reward = valid_rentals * self.rental_reward
                        next_n = min(self.max_bikes, curr_n1 - valid_rentals + ret)
                        probs_1[next_n] += prob
                        rewards_1 += prob * reward
                        
                # Loc 2 transitions
                probs_2 = np.zeros(self.max_bikes + 1)
                rewards_2 = 0
                
                for req in range(upper_bound):
                    for ret in range(upper_bound):
                        prob = p_req2[req] * p_ret2[ret]
                        valid_rentals = min(curr_n2, req)
                        reward = valid_rentals * self.rental_reward
                        next_n = min(self.max_bikes, curr_n2 - valid_rentals + ret)
                        probs_2[next_n] += prob
                        rewards_2 += prob * reward
                
                # Combine
                # P(s'|s,a) = P(n1'|...) * P(n2'|...)
                # R(s,a) = cost + E[R1] + E[R2]
                
                self.R[s_idx, a_idx] = expected_reward + rewards_1 + rewards_2
                
                # Outer product for joint probabilities
                # This fills the row in P
                # P[s_idx, a_idx, next_s_idx]
                # next_s_idx corresponds to (next_n1, next_n2)
                # We can iterate to fill
                for i in range(self.max_bikes + 1):
                    if probs_1[i] == 0: continue
                    for j in range(self.max_bikes + 1):
                        if probs_2[j] == 0: continue
                        
                        next_s_idx = self.state_to_idx[(i, j)]
                        self.P[s_idx, a_idx, next_s_idx] = probs_1[i] * probs_2[j]

    def solve(self):
        iterations = 0
        while True:
            # Policy Evaluation
            # V = R_pi + gamma * P_pi * V
            # (I - gamma * P_pi) * V = R_pi
            
            # Construct P_pi and R_pi
            # policy is array of action indices
            
            # Advanced indexing
            row_indices = np.arange(self.num_states)
            action_indices = self.policy
            
            P_pi = self.P[row_indices, action_indices, :]
            R_pi = self.R[row_indices, action_indices]
            
            # Solve linear system
            A = np.eye(self.num_states) - self.discount * P_pi
            self.V = np.linalg.solve(A, R_pi)
            
            # Policy Improvement
            policy_stable = True
            
            # Q = R + gamma * P * V
            # Q shape: (num_states, num_actions)
            # V shape: (num_states,)
            # P shape: (num_states, num_actions, num_states)
            
            # Broadcast V: (1, 1, num_states) -> (num_states, num_actions, num_states) ?
            # P @ V -> (num_states, num_actions)
            
            PV = np.einsum('san,n->sa', self.P, self.V)
            Q = self.R + self.discount * PV
            
            new_policy = np.argmax(Q, axis=1)
            
            if not np.array_equal(self.policy, new_policy):
                policy_stable = False
                self.policy = new_policy
            
            print(f"Iteration {iterations}, Stable: {policy_stable}")
            if policy_stable:
                break
            iterations += 1
            
        return self.policy, self.V

    def plot_policy(self, title="Policy", filename=None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.policy_grid(), cmap="coolwarm", annot=True, fmt="d", center=0)
        plt.title(title)
        plt.xlabel("Bikes at Location 2")
        plt.ylabel("Bikes at Location 1")
        plt.gca().invert_yaxis() # 0 at bottom
        if filename:
            plt.savefig(filename)
            print(f"Saved plot to {filename}")
        else:
            plt.show()

    def plot_value(self, title="Value Function", filename=None):
        plt.figure(figsize=(10, 8))
        # Reshape V to grid
        V_grid = self.V.reshape((self.max_bikes + 1, self.max_bikes + 1))
        sns.heatmap(V_grid, cmap="viridis", annot=False) # Annotating values might be too crowded
        plt.title(title)
        plt.xlabel("Bikes at Location 2")
        plt.ylabel("Bikes at Location 1")
        plt.gca().invert_yaxis()
        if filename:
            plt.savefig(filename)
            print(f"Saved plot to {filename}")
        else:
            plt.show()
            
    def policy_grid(self):
        grid = np.zeros((self.max_bikes + 1, self.max_bikes + 1), dtype=int)
        for s_idx, action_idx in enumerate(self.policy):
            r, c = self.states[s_idx]
            grid[r, c] = self.actions[action_idx]
        return grid

    def print_policy(self):
        print("Policy (Action: Net moved 1->2):")
        grid = self.policy_grid()
            
        for i in range(self.max_bikes, -1, -1):
            row = []
            for j in range(self.max_bikes + 1):
                row.append(f"{grid[i, j]:2d}")
            print(" ".join(row))

if __name__ == "__main__":
    start_time = time.time()
    print("Solving Original Problem...")
    gbike = GbikeRental(modified=False)
    policy, V = gbike.solve()
    gbike.print_policy()
    gbike.plot_policy(title="Original Policy", filename="policy_original.png")
    gbike.plot_value(title="Original Value Function", filename="value_original.png")
    
    print(f"\nSolving Modified Problem... (Time elapsed: {time.time() - start_time:.2f}s)")
    gbike_mod = GbikeRental(modified=True)
    policy_mod, V_mod = gbike_mod.solve()
    gbike_mod.print_policy()
    gbike_mod.plot_policy(title="Modified Policy (Free Shuttle + Parking Cost)", filename="policy_modified.png")
    gbike_mod.plot_value(title="Modified Value Function", filename="value_modified.png")
    print(f"Total Time: {time.time() - start_time:.2f}s")
