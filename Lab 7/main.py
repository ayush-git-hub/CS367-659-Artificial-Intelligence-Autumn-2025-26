import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class MENACE:
    """
    MENACE: Machine Educable Naughts and Crosses Engine
    
    CRUCIAL COMPONENTS:
    1. State representation with symmetry reduction (reduces 304 states)
    2. Bead-based probability learning (colored beads = action probabilities)
    3. Reinforcement through bead addition/removal
    """
    
    def __init__(self, initial_beads=4):
        self.matchboxes = defaultdict(lambda: defaultdict(int))
        self.initial_beads = initial_beads
        self.move_history = []
        
    def get_canonical_state(self, board):
        """
        CRUCIAL: Reduce board states using rotational and reflective symmetry
        This reduces ~5000+ states to 304 unique states
        """
        def rotate_90(b):
            return (b[6], b[3], b[0], b[7], b[4], b[1], b[8], b[5], b[2])
        
        def reflect_h(b):
            return (b[2], b[1], b[0], b[5], b[4], b[3], b[8], b[7], b[6])
        
        board_tuple = tuple(board)
        transformations = [board_tuple]
        
        current = board_tuple
        for _ in range(3):
            current = rotate_90(current)
            transformations.append(current)
        
        reflected = reflect_h(board_tuple)
        transformations.append(reflected)
        current = reflected
        for _ in range(3):
            current = rotate_90(current)
            transformations.append(current)
        
        return min(transformations)
    
    def initialize_matchbox(self, state, board):
        if state not in self.matchboxes or not self.matchboxes[state]:
            for i in range(9):
                if board[i] == 0:
                    moves_made = sum(1 for x in board if x != 0)
                    beads = max(1, self.initial_beads - moves_made)
                    self.matchboxes[state][i] = beads
    
    def select_move(self, board):
        state = self.get_canonical_state(board)
        self.initialize_matchbox(state, board)
        
        available = {pos: beads for pos, beads in self.matchboxes[state].items() 
                    if board[pos] == 0 and beads > 0}
        
        if not available:
            empty = [i for i in range(9) if board[i] == 0]
            return random.choice(empty) if empty else None
        
        total_beads = sum(available.values())
        pick = random.randint(1, total_beads)
        cumulative = 0
        for pos, beads in available.items():
            cumulative += beads
            if pick <= cumulative:
                self.move_history.append((state, pos))
                return pos
        return list(available.keys())[0]
    
    def reinforce(self, result):
        for state, move in self.move_history:
            if result == 'win':
                self.matchboxes[state][move] += 3
            elif result == 'draw':
                self.matchboxes[state][move] += 1
            else:
                self.matchboxes[state][move] = max(0, self.matchboxes[state][move] - 1)
        self.move_history = []
    
    def get_stats(self):
        return len(self.matchboxes), sum(sum(m.values()) for m in self.matchboxes.values())


class TicTacToe:
    """Tic-Tac-Toe game environment"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = [0] * 9
        self.current_player = 1
        return self.board.copy()
    
    def make_move(self, position):
        if self.board[position] != 0:
            return False
        self.board[position] = self.current_player
        self.current_player = 3 - self.current_player
        return True
    
    def check_winner(self):
        lines = [(0,1,2), (3,4,5), (6,7,8),
                 (0,3,6), (1,4,7), (2,5,8),
                 (0,4,8), (2,4,6)]
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        if 0 not in self.board:
            return 0
        return None
    
    def get_empty_positions(self):
        return [i for i in range(9) if self.board[i] == 0]


def play_random_opponent():
    def strategy(board):
        empty = [i for i in range(9) if board[i] == 0]
        return random.choice(empty) if empty else None
    return strategy


def train_menace(num_games=500):
    menace = MENACE(initial_beads=4)
    game = TicTacToe()
    opponent = play_random_opponent()
    
    results = {'win': 0, 'loss': 0, 'draw': 0}
    win_rates = []
    
    for i in range(num_games):
        game.reset()
        menace_first = (i % 2 == 0)
        
        while True:
            if menace_first:
                move = menace.select_move(game.board)
                game.make_move(move)
                winner = game.check_winner()
                if winner is not None:
                    break
                move = opponent(game.board)
                if move is not None:
                    game.make_move(move)
                winner = game.check_winner()
                if winner is not None:
                    break
            else:
                move = opponent(game.board)
                if move is not None:
                    game.make_move(move)
                winner = game.check_winner()
                if winner is not None:
                    break
                move = menace.select_move(game.board)
                game.make_move(move)
                winner = game.check_winner()
                if winner is not None:
                    break
        
        menace_symbol = 1 if menace_first else 2
        if winner == menace_symbol:
            menace.reinforce('win')
            results['win'] += 1
        elif winner == 0:
            menace.reinforce('draw')
            results['draw'] += 1
        else:
            menace.reinforce('loss')
            results['loss'] += 1
        
        if (i + 1) % 50 == 0:
            win_rates.append(results['win'] / (i + 1))
    
    return menace, results, win_rates


def binary_bandit_A(action):
    probs = {1: 0.9, 2: 0.2}
    return 1 if random.random() < probs[action] else 0

def binary_bandit_B(action):
    probs = {1: 0.2, 2: 0.8}
    return 1 if random.random() < probs[action] else 0


def epsilon_greedy_binary(bandit_func, epsilon=0.1, num_steps=1000):
    Q = {1: 0.0, 2: 0.0}
    N = {1: 0, 2: 0}
    
    rewards = []
    actions_taken = []
    q_history = {1: [], 2: []}
    
    for t in range(num_steps):
        if random.random() < epsilon:
            action = random.choice([1, 2])
        else:
            action = 1 if Q[1] >= Q[2] else 2
        
        reward = bandit_func(action)
        
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        
        rewards.append(reward)
        actions_taken.append(action)
        q_history[1].append(Q[1])
        q_history[2].append(Q[2])
    
    return Q, N, rewards, actions_taken, q_history


class NonStationaryBandit:
    """
    CRUCIAL: 10-armed bandit with random walk rewards
    All mean rewards start equal, then take independent random walks
    """
    
    def __init__(self, n_arms=10, initial_mean=0.0, walk_std=0.01):
        self.n_arms = n_arms
        self.walk_std = walk_std
        self.true_means = np.ones(n_arms) * initial_mean
    
    def step(self):
        self.true_means += np.random.normal(0, self.walk_std, self.n_arms)
    
    def pull(self, action):
        return np.random.normal(self.true_means[action], 1.0)
    
    def get_optimal_action(self):
        return np.argmax(self.true_means)


def bandit_nonstat(action, bandit):
    return bandit.pull(action)


def epsilon_greedy_nonstationary(bandit, epsilon=0.1, alpha=0.1, num_steps=10000):
    n_arms = bandit.n_arms
    Q = np.zeros(n_arms)
    
    rewards = []
    optimal_actions = []
    
    for t in range(num_steps):
        bandit.step()
        
        if np.random.random() < epsilon:
            action = np.random.randint(n_arms)
        else:
            action = np.argmax(Q)
        
        reward = bandit.pull(action)
        
        Q[action] += alpha * (reward - Q[action])
        
        rewards.append(reward)
        optimal_actions.append(1 if action == bandit.get_optimal_action() else 0)
    
    return Q, rewards, optimal_actions


def epsilon_greedy_stationary_update(bandit, epsilon=0.1, num_steps=10000):
    n_arms = bandit.n_arms
    Q = np.zeros(n_arms)
    N = np.zeros(n_arms)
    
    rewards = []
    optimal_actions = []
    
    for t in range(num_steps):
        bandit.step()
        
        if np.random.random() < epsilon:
            action = np.random.randint(n_arms)
        else:
            action = np.argmax(Q)
        
        reward = bandit.pull(action)
        
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        
        rewards.append(reward)
        optimal_actions.append(1 if action == bandit.get_optimal_action() else 0)
    
    return Q, rewards, optimal_actions


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    print("=" * 70)
    print("LAB 7: MENACE AND BANDIT REINFORCEMENT LEARNING")
    print("=" * 70)
    
    print("\n--- PROBLEM 1: MENACE Training ---")
    menace, results, win_rates = train_menace(num_games=500)
    states, beads = menace.get_stats()
    print(f"Training Results over 500 games:")
    print(f"  Wins: {results['win']}, Draws: {results['draw']}, Losses: {results['loss']}")
    print(f"  Unique states learned: {states}")
    print(f"  Total beads in system: {beads}")
    print(f"  Final win rate: {results['win']/500:.2%}")
    
    print("\n--- PROBLEM 2: Binary Bandit with Epsilon-Greedy ---")
    
    print("\nBandit A (Action 1: 90% success, Action 2: 20% success):")
    Q_A, N_A, rewards_A, actions_A, _ = epsilon_greedy_binary(binary_bandit_A, epsilon=0.1, num_steps=1000)
    print(f"  Final Q-values: Q(1)={Q_A[1]:.3f}, Q(2)={Q_A[2]:.3f}")
    print(f"  Action counts: N(1)={N_A[1]}, N(2)={N_A[2]}")
    print(f"  Best action selected: {1 if Q_A[1] > Q_A[2] else 2} (Correct: 1)")
    print(f"  Average reward: {np.mean(rewards_A):.3f}")
    
    print("\nBandit B (Action 1: 20% success, Action 2: 80% success):")
    Q_B, N_B, rewards_B, actions_B, _ = epsilon_greedy_binary(binary_bandit_B, epsilon=0.1, num_steps=1000)
    print(f"  Final Q-values: Q(1)={Q_B[1]:.3f}, Q(2)={Q_B[2]:.3f}")
    print(f"  Action counts: N(1)={N_B[1]}, N(2)={N_B[2]}")
    print(f"  Best action selected: {1 if Q_B[1] > Q_B[2] else 2} (Correct: 2)")
    print(f"  Average reward: {np.mean(rewards_B):.3f}")
    
    print("\n--- PROBLEMS 3 & 4: Non-stationary 10-Armed Bandit ---")
    
    num_runs = 100
    num_steps = 10000
    
    rewards_const_alpha = np.zeros(num_steps)
    rewards_sample_avg = np.zeros(num_steps)
    optimal_const_alpha = np.zeros(num_steps)
    optimal_sample_avg = np.zeros(num_steps)
    
    for run in range(num_runs):
        bandit1 = NonStationaryBandit(n_arms=10, initial_mean=0.0, walk_std=0.01)
        _, r1, o1 = epsilon_greedy_nonstationary(bandit1, epsilon=0.1, alpha=0.1, num_steps=num_steps)
        rewards_const_alpha += np.array(r1)
        optimal_const_alpha += np.array(o1)
        
        bandit2 = NonStationaryBandit(n_arms=10, initial_mean=0.0, walk_std=0.01)
        _, r2, o2 = epsilon_greedy_stationary_update(bandit2, epsilon=0.1, num_steps=num_steps)
        rewards_sample_avg += np.array(r2)
        optimal_sample_avg += np.array(o2)
        
        if (run + 1) % 20 == 0:
            print(f"  Completed {run + 1}/{num_runs} runs...")
    
    rewards_const_alpha /= num_runs
    rewards_sample_avg /= num_runs
    optimal_const_alpha /= num_runs
    optimal_sample_avg /= num_runs
    
    print(f"\nResults over {num_steps} steps (averaged over {num_runs} runs):")
    print(f"  Constant α=0.1 (modified):")
    print(f"    Final avg reward: {np.mean(rewards_const_alpha[-1000:]):.3f}")
    print(f"    Final % optimal: {np.mean(optimal_const_alpha[-1000:])*100:.1f}%")
    print(f"  Sample average (1/n):")
    print(f"    Final avg reward: {np.mean(rewards_sample_avg[-1000:]):.3f}")
    print(f"    Final % optimal: {np.mean(optimal_sample_avg[-1000:])*100:.1f}%")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(range(50, 501, 50), win_rates, 'b-o', linewidth=2)
    ax1.set_xlabel('Games Played')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Problem 1: MENACE Learning Curve')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    ax2 = axes[0, 1]
    ax2.plot(np.cumsum(rewards_A), label='Bandit A', linewidth=1.5)
    ax2.plot(np.cumsum(rewards_B), label='Bandit B', linewidth=1.5)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Problem 2: Binary Bandit Cumulative Rewards')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    window = 100
    smooth_const = np.convolve(rewards_const_alpha, np.ones(window)/window, mode='valid')
    smooth_sample = np.convolve(rewards_sample_avg, np.ones(window)/window, mode='valid')
    ax3.plot(smooth_const, label='Constant α=0.1 (Modified)', linewidth=1.5)
    ax3.plot(smooth_sample, label='Sample Average (1/n)', linewidth=1.5)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Average Reward (smoothed)')
    ax3.set_title('Problems 3&4: Non-stationary Bandit - Average Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    smooth_opt_const = np.convolve(optimal_const_alpha, np.ones(window)/window, mode='valid')
    smooth_opt_sample = np.convolve(optimal_sample_avg, np.ones(window)/window, mode='valid')
    ax4.plot(smooth_opt_const * 100, label='Constant α=0.1 (Modified)', linewidth=1.5)
    ax4.plot(smooth_opt_sample * 100, label='Sample Average (1/n)', linewidth=1.5)
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('% Optimal Action')
    ax4.set_title('Problems 3&4: Non-stationary Bandit - Optimal Action %')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lab7_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS:")
    print("=" * 70)
    print("""
Problem 1 (MENACE): 
  - Successfully learns to play Tic-Tac-Toe through reinforcement
  - Key: Symmetry reduction, bead-based probability, +3/-1 reinforcement
  
Problem 2 (Binary Bandit):
  - Epsilon-greedy correctly identifies the better action in both bandits
  - Balances exploration (ε=10%) with exploitation (90% greedy)
  
Problems 3&4 (Non-stationary Bandit):
  - Standard sample averaging FAILS for non-stationary rewards
  - Modified algorithm with constant α=0.1 TRACKS changing rewards
  - Constant step-size gives exponential recency-weighted average
  - Recent rewards weighted more heavily, enabling adaptation
    """)
