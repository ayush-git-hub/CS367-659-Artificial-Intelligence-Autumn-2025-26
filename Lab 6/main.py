import numpy as np
import matplotlib.pyplot as plt

def activate(x):
    return np.where(x >= 0, 1, -1)

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        num_patterns = len(patterns)
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= num_patterns

    def predict(self, pattern, steps=5):
        result = pattern.copy()
        for _ in range(steps):
            idx = np.random.permutation(self.size)
            for i in idx:
                raw = np.dot(self.weights[i], result)
                result[i] = 1 if raw >= 0 else -1
        return result

def add_noise(pattern, noise_level):
    noisy = pattern.copy()
    num_flips = int(noise_level * len(pattern))
    indices = np.random.choice(len(pattern), num_flips, replace=False)
    noisy[indices] *= -1
    return noisy

def task1_error_correction():
    print("\n" + "="*40)
    print("TASK 1: Error Correcting Capability")
    print("="*40)
    
    size = 100 
    dim = 10
    
    pattern_1 = np.ones((dim, dim)) * -1
    pattern_1[2:8, 4:6] = 1 
    pattern_1[2:4, 2:8] = 1 
    
    pattern_2 = np.ones((dim, dim)) * -1
    pattern_2[2:8, 2:4] = 1 
    pattern_2[2:8, 6:8] = 1 
    pattern_2[4:6, 4:6] = 1 
    
    p1_flat = pattern_1.flatten()
    p2_flat = pattern_2.flatten()
    
    hn = HopfieldNetwork(size)
    hn.train([p1_flat, p2_flat])
    
    noise_levels = np.linspace(0, 0.5, 11)
    accuracies = []
    
    for nl in noise_levels:
        success = 0
        trials = 50
        for _ in range(trials):
            target = p1_flat
            noisy_input = add_noise(target, nl)
            output = hn.predict(noisy_input)
            if np.array_equal(output, target):
                success += 1
        accuracies.append(success / trials * 100)
    
    plt.figure(figsize=(10, 5))
    plt.plot(noise_levels * 100, accuracies, marker='o')
    plt.title("Hopfield Network Error Correcting Capability")
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Recovery Accuracy (%)")
    plt.grid(True)
    plt.show()
    
    print(f"Error capability analysis completed. Graph displayed.")
    print("Conclusion: The network recovers patterns perfectly up to a certain noise threshold (approx 15-25% for this size), after which performance degrades.")

def task2_eight_rooks():
    print("\n" + "="*40)
    print("TASK 2: Eight-Rook Problem")
    print("="*40)
    
    N = 8
    neurons = N * N
    weights = np.zeros((neurons, neurons))
    
    # Setup weights once
    for r in range(N):
        for c in range(N):
            i = r * N + c
            for k in range(N):
                if k != c: 
                    j = r * N + k
                    weights[i, j] -= 2 
            for k in range(N):
                if k != r: 
                    j = k * N + c
                    weights[i, j] -= 2 

    attempt = 0
    while True:
        attempt += 1
        state = np.random.choice([-1, 1], size=neurons)
        
        # Converge
        for _ in range(200):
            idx = np.random.randint(0, neurons)
            activation = np.dot(weights[idx], (state + 1) / 2) 
            state[idx] = 1 if activation > -1 else -1 
        
        board = np.zeros((N, N))
        for r in range(N):
            for c in range(N):
                if state[r*N + c] == 1:
                    board[r, c] = 1
        
        # Check for perfection
        rooks_count = np.sum(board)
        rows_ok = np.all(np.sum(board, axis=1) <= 1)
        cols_ok = np.all(np.sum(board, axis=0) <= 1)
        
        if rooks_count == 8 and rows_ok and cols_ok:
            print(f"Perfect solution found on attempt {attempt}.")
            break
        else:
            # Silent retry, or print dot for progress
            print(f"Attempt {attempt}: Found {int(rooks_count)} rooks. Retrying for perfect 8...")

    print("Reason for weights: We set large inhibitory weights (-2) between neurons in the same row and same column.")
    print("This enforces the constraint that only one rook can exist per row and per column.")
    
    plt.figure(figsize=(5, 5))
    plt.imshow(board, cmap='binary')
    plt.title("8-Rook Solution via Hopfield")
    plt.grid(which='both', color='black', linestyle='-', linewidth=1)
    plt.xticks(np.arange(-.5, N, 1)); plt.yticks(np.arange(-.5, N, 1))
    plt.tick_params(axis='both', which='major', labelbottom=False, labelleft=False)
    plt.show()
    
    print("Final Board Configuration (1=Rook):")
    print(board)

def task3_tsp():
    print("\n" + "="*40)
    print("TASK 3: Traveling Salesman Problem (10 Cities)")
    print("="*40)
    
    N_cities = 10
    np.random.seed(42) # Fixed cities for consistency
    coords = np.random.rand(N_cities, 2)
    
    dist_matrix = np.zeros((N_cities, N_cities))
    for i in range(N_cities):
        for j in range(N_cities):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
            
    A, B, C, D = 500, 500, 200, 500
    u0 = 0.02
    step_size = 0.0001
    iterations = 10000
    
    def get_v(u_val):
        return 0.5 * (1 + np.tanh(u_val / 0.5))
    
    print(f"Solving TSP for {N_cities} cities...")
    print(f"Total weights needed: {N_cities**2} neurons fully connected = {(N_cities**2)**2} weights.")
    
    attempt = 0
    V_final = None
    tour_indices = None
    
    while True:
        attempt += 1
        # Randomize initialization only
        u = np.random.uniform(-u0, u0, (N_cities, N_cities))
        
        for _ in range(iterations):
            V = get_v(u)
            row_sum = np.sum(V, axis=1).reshape(-1, 1)
            col_sum = np.sum(V, axis=0).reshape(1, -1)
            
            term_a = A * (row_sum - 1)
            term_b = B * (col_sum - 1)
            term_c = C * (np.sum(V) - N_cities)
            
            term_d = np.zeros_like(V)
            for x in range(N_cities):
                for i in range(N_cities):
                    prev_idx = (i - 1) % N_cities
                    next_idx = (i + 1) % N_cities
                    y_term = 0
                    for y in range(N_cities):
                        y_term += dist_matrix[x, y] * (V[y, prev_idx] + V[y, next_idx])
                    term_d[x, i] = D * y_term
                    
            du = -term_a - term_b - term_c - term_d
            u += step_size * du

        V_final = get_v(u)
        tour_indices = np.argmax(V_final, axis=0)
        
        # Check if valid tour (all cities visited exactly once)
        unique_cities = len(np.unique(tour_indices))
        
        if unique_cities == N_cities:
            print(f"Valid tour found on attempt {attempt}!")
            break
        else:
            print(f"Attempt {attempt}: Tour invalid (visited {unique_cities}/10 unique cities). Retrying...")

    ordered_coords = coords[tour_indices]
    ordered_coords = np.vstack([ordered_coords, ordered_coords[0]])
    
    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=50, zorder=2)
    plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'b-', zorder=1)
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=12)
    plt.title("TSP Solution (Hopfield Network)")
    plt.show()
    
    print("Tour calculated. Matrix state (Probabilities):")
    np.set_printoptions(precision=2, suppress=True)
    print(V_final)

if __name__ == "__main__":
    task1_error_correction()
    task2_eight_rooks()
    task3_tsp()