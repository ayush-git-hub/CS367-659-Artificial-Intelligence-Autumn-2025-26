import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import permutations

"""
Ultra-Optimized Jigsaw Solver with Corner/Edge Heuristics
"""

random.seed(42)
np.random.seed(42)

"""
1. Load Octave ASCII Image
"""
def load_octave_ascii_image(filename):
    print(f"[INFO] Loading file: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data_lines = []
    dims = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if not stripped or stripped.startswith('#'):
            continue
        
        if dims is None:
            try:
                dims = list(map(int, stripped.split()))
                print(f"[INFO] Image dimensions: {dims[0]}x{dims[1]}")
            except ValueError:
                continue
        else:
            try:
                data_lines.append(int(stripped))
            except ValueError:
                continue

    if dims is None:
        raise ValueError("Could not read image dimensions")
    
    height, width = dims
    
    if len(data_lines) != height * width:
        raise ValueError(f"Pixel count mismatch")

    image = np.array(data_lines, dtype=np.uint8).reshape((height, width))
    return image

"""
2. Load and Split Image
"""
filename = 'scrambled_lena_github.mat'
scrambled_image = load_octave_ascii_image(filename)
scrambled_image = np.rot90(scrambled_image, k=-1)

H, W = scrambled_image.shape
tile_size = 128
pieces = []

for i in range(4):
    for j in range(4):
        piece = scrambled_image[i*tile_size:(i+1)*tile_size,
                                j*tile_size:(j+1)*tile_size]
        pieces.append(piece)

n = len(pieces)
print(f"[INFO] Loaded {n} pieces")

"""
3. Compatibility Matrices
"""
def compute_compatibility_matrices():
    right_compat = np.zeros((n, n))
    bottom_compat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                diff_r = pieces[i][:, -1].astype(float) - pieces[j][:, 0].astype(float)
                right_compat[i, j] = np.sum(diff_r ** 2)

                diff_b = pieces[i][-1, :].astype(float) - pieces[j][ 0, :].astype(float)
                bottom_compat[i, j] = np.sum(diff_b ** 2)
            else:
                right_compat[i, j] = float('inf')
                bottom_compat[i, j] = float('inf')

    return right_compat, bottom_compat

print("[INFO] Computing compatibility matrices...")
right_compat, bottom_compat = compute_compatibility_matrices()

"""
4. Energy Function
"""
def compute_energy(state):
    total = 0
    for i in range(4):
        for j in range(4):
            pos = i * 4 + j
            piece = state[pos]

            if j < 3:
                right_piece = state[i * 4 + j + 1]
                total += right_compat[piece, right_piece]

            if i < 3:
                bottom_piece = state[(i + 1) * 4 + j]
                total += bottom_compat[piece, bottom_piece]

    return total

"""
5. Advanced Greedy Row-by-Row Construction
"""
def advanced_greedy_construction():
    print("[INFO] Running advanced greedy construction...")
    state = [-1] * 16
    used = set()
    
    # Try multiple starting pieces and pick best
    best_full_state = None
    best_full_energy = float('inf')
    
    for start_piece in range(n):
        temp_state = [-1] * 16
        temp_used = set()
        temp_state[0] = start_piece
        temp_used.add(start_piece)
        
        # Fill the  rest greedily
        for pos in range(1, 16):
            i, j = divmod(pos, 4)
            
            best_piece = None
            best_cost = float('inf')
            
            for piece in range(n):
                if piece in temp_used:
                    continue
                
                cost = 0
                count = 0
                
                if j > 0:  # Left neighbor
                    left_piece = temp_state[i * 4 + j - 1]
                    cost += right_compat[left_piece, piece]
                    count += 1
                
                if i > 0:  # Top neighbor
                    top_piece = temp_state[(i - 1) * 4 + j]
                    cost += bottom_compat[top_piece, piece]
                    count += 1
                
                if count > 0:
                    cost /= count  # Average cost
                
                if cost < best_cost:
                    best_cost = cost
                    best_piece = piece
            
            temp_state[pos] = best_piece
            temp_used.add(best_piece)
        
        temp_energy = compute_energy(temp_state)
        if temp_energy < best_full_energy:
            best_full_energy = temp_energy
            best_full_state = temp_state[:]
    
    print(f"[INFO] Greedy construction energy: {best_full_energy:.0f}")
    return best_full_state

"""
6. Ultra-Aggressive Simulated Annealing
"""
def ultra_aggressive_sa(initial_state, max_iter=100000):
    current = initial_state[:]
    best = current[:]
    current_energy = compute_energy(current)
    best_energy = current_energy
    
    T = 50000.0
    alpha = 0.99998
    
    for iteration in range(max_iter):
        # Try both 2-swap and 3-swap
        if random.random() < 0.8:
            # 2-swap
            i, j = random.sample(range(n), 2)
            neighbor = current[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # 3-swap (circular shift)
            i, j, k = random.sample(range(n), 3)
            neighbor = current[:]
            neighbor[i], neighbor[j], neighbor[k] = neighbor[k], neighbor[i], neighbor[j]
        
        neighbor_energy = compute_energy(neighbor)
        delta = neighbor_energy - current_energy
        
        if delta <= 0 or (T > 0 and random.random() < math.exp(min(0, -delta / T))):
            current = neighbor
            current_energy = neighbor_energy
            
            if neighbor_energy < best_energy:
                best = neighbor[:]
                best_energy = neighbor_energy
        
        T *= alpha
        
        if iteration % 10000 == 0 and iteration > 0:
            print(f"Iter {iteration:6d} | Best: {best_energy:10.0f} | Temp: {T:8.1f}")
    
    return best, best_energy

"""
7. Aggressive Local Search
"""
def aggressive_local_search(state, max_iter=20000):
    print("[INFO] Running aggressive local search...")
    current = state[:]
    current_energy = compute_energy(current)
    
    improvements = 0
    for iteration in range(max_iter):
        improved = False
        
        # Try all pairs
        for i in range(n):
            for j in range(i + 1, n):
                neighbor = current[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbor_energy = compute_energy(neighbor)
                
                if neighbor_energy < current_energy:
                    current = neighbor
                    current_energy = neighbor_energy
                    improved = True
                    improvements += 1
                    break
            if improved:
                break
        
        if not improved:
            break
    
    print(f"[INFO] Local search: {improvements} improvements, final energy: {current_energy:.0f}")
    return current, current_energy

"""
8. Main Solver with Multiple Strategies
"""
def run_ultimate_solver(num_attempts=50):
    print(f"\n[INFO] Running ultimate solver with {num_attempts} attempts...")
    
    global_best_state = None
    global_best_energy = float('inf')
    
    for attempt in range(num_attempts):
        print(f"\n{'='*60}")
        print(f"ATTEMPT {attempt + 1}/{num_attempts}")
        print(f"{'='*60}")
        
        if attempt < num_attempts // 2:
            # Use greedy construction
            initial = advanced_greedy_construction()
        else:
            # Random initialization
            initial = list(range(n))
            random.shuffle(initial)
            init_energy = compute_energy(initial)
            print(f"[INFO] Random initialization energy: {init_energy:.0f}")
        
        # Apply SA
        print("[INFO] Running SA...")
        sa_state, sa_energy = ultra_aggressive_sa(initial)
        print(f"[INFO] SA result: {sa_energy:.0f}")
        
        # Local search refinement
        final_state, final_energy = aggressive_local_search(sa_state)
        
        if final_energy < global_best_energy:
            global_best_state = final_state
            global_best_energy = final_energy  
            print(f"[INFO] *** NEW GLOBAL BEST: {final_energy:.0f} ***")
    
    return global_best_state, global_best_energy

"""
9. Execute
"""
print("\n" + "=" * 70)
print("ULTIMATE JIGSAW PUZZLE SOLVER")
print("=" * 70)

try:
    solution, final_energy = run_ultimate_solver(num_attempts=50)
except KeyboardInterrupt:
    print("\n[INFO] Interrupted! Using greedy solution...")
    solution = advanced_greedy_construction()
    final_energy = compute_energy(solution)

print(f"\n{'='*70}")
print(f"FINAL SOLUTION")
print(f"{'='*70}")
print(f"Energy: {final_energy:.0f}")
print(f"State: {solution}")

"""
10. Visualize
"""
def reconstruct(state):
    canvas = np.zeros((512, 512), dtype=np.uint8)
    for idx in range(16):
        i, j = divmod(idx, 4)
        piece_idx = state[idx]
        canvas[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = pieces[piece_idx]
    return canvas

reconstructed = reconstruct(solution)
scrambled = reconstruct(list(range(n)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(scrambled, cmap='gray')
axes[0].set_title("Scrambled", fontsize=16, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(reconstructed, cmap='gray')
axes[1].set_title(f"Solved (Energy={final_energy:.0f})", fontsize=16, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig("results/jigsaw_perfect.png", dpi=200, bbox_inches='tight')
print("\n[INFO] Saved to 'jigsaw_perfect.png'")
plt.show()
