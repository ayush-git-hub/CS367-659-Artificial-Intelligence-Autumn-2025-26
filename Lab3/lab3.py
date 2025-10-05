import random
import time
import itertools
from typing import List, Tuple, Callable
from tabulate import tabulate  

def generate_ksat(k: int, m: int, n: int) -> List[List[int]]:
    clauses = []
    for _ in range(m):
        # Pick k distinct variables
        vars_selected = random.sample(range(1, n + 1), k)
        clause = []
        for var in vars_selected:
            # Random sign: + for positive, - for negation
            sign = 1 if random.random() < 0.5 else -1
            clause.append(sign * var)
        # Shuffle for randomness
        random.shuffle(clause)
        clauses.append(clause)
    return clauses

def evaluate(assignment: List[int], clauses: List[List[int]]) -> int:
    
    satisfied = 0
    for clause in clauses:
        clause_satisfied = False
        for lit in clause:
            var_idx = abs(lit) - 1  # 0-based index
            var_val = assignment[var_idx]
            lit_val = var_val if lit > 0 else 1 - var_val
            if lit_val == 1:
                clause_satisfied = True
                break
        if clause_satisfied:
            satisfied += 1
    return satisfied

def heuristic1(assignment: List[int], clauses: List[List[int]]) -> float:
 
    return float(evaluate(assignment, clauses))

def heuristic2(assignment: List[int], clauses: List[List[int]]) -> float:

    satisfied = 0
    almost_satisfied = 0
    for clause in clauses:
        true_count = 0
        for lit in clause:
            var_idx = abs(lit) - 1
            var_val = assignment[var_idx]
            lit_val = var_val if lit > 0 else 1 - var_val
            if lit_val == 1:
                true_count += 1
        if true_count > 0:
            satisfied += 1
            if true_count == 1:
                almost_satisfied += 1
    return satisfied + 0.5 * almost_satisfied

def hill_climbing(clauses: List[List[int]], n: int, heuristic_func: Callable, max_iters: int = 1000) -> Tuple[bool, int, float, int]:

    assignment = [random.randint(0, 1) for _ in range(n)]
    start_time = time.time()
    iters = 0
    nodes_explored = 1  # Count initial state
    
    while iters < max_iters:
        current_h = heuristic_func(assignment, clauses)
        
        # Check if solution found
        if current_h >= len(clauses):
            break
            
        best_assign = assignment[:]
        best_h = current_h
        
        # Single-flip neighbors
        for i in range(n):
            new_assign = assignment[:]
            new_assign[i] = 1 - new_assign[i]
            new_h = heuristic_func(new_assign, clauses)
            nodes_explored += 1  # Count each neighbor evaluated
            if new_h > best_h:
                best_assign = new_assign
                best_h = new_h
        
        if best_h > current_h:
            assignment = best_assign
        else:
            break  # Local optimum
        iters += 1
    
    elapsed = time.time() - start_time
    success = evaluate(assignment, clauses) == len(clauses)
    return success, iters, elapsed, nodes_explored

def beam_search(clauses: List[List[int]], n: int, beam_width: int, heuristic_func: Callable, max_iters: int = 1000) -> Tuple[bool, int, float, int]:

    # Initialize beam with random states
    beam = [[random.randint(0, 1) for _ in range(n)] for _ in range(beam_width)]
    start_time = time.time()
    iters = 0
    nodes_explored = beam_width  # Count initial beam states
    
    while iters < max_iters:
        # Check if any in beam is solution
        if any(evaluate(s, clauses) == len(clauses) for s in beam):
            break
            
        new_beam = []
        for state in beam:
            # Generate single-flip neighbors
            for i in range(n):
                new_state = state[:]
                new_state[i] = 1 - new_state[i]
                new_beam.append(new_state)
                nodes_explored += 1  # Count each generated neighbor
        
        # Remove duplicates by converting to tuples
        unique_beam = list({tuple(s): s for s in new_beam}.values())
        
        # Sort by heuristic, keep top beam_width
        unique_beam.sort(key=lambda s: heuristic_func(s, clauses), reverse=True)
        beam = unique_beam[:beam_width]
        
        iters += 1
    
    elapsed = time.time() - start_time
    best_state = max(beam, key=lambda s: heuristic_func(s, clauses))
    success = evaluate(best_state, clauses) == len(clauses)
    return success, iters, elapsed, nodes_explored

def vnd(clauses: List[List[int]], n: int, heuristic_func: Callable, max_iters: int = 1000) -> Tuple[bool, int, float, int]:

    assignment = [random.randint(0, 1) for _ in range(n)]
    start_time = time.time()
    iters = 0
    nodes_explored = 1  # Count initial state
    neighborhoods = ["single", "double", "targeted"]
    neigh_idx = 0
    
    while iters < max_iters:
        current_h = heuristic_func(assignment, clauses)
        
        # Check if solution found
        if evaluate(assignment, clauses) == len(clauses):
            break
        
        improved = False
        neigh_type = neighborhoods[neigh_idx]
        best_h = current_h
        best_assign = assignment[:]
        
        if neigh_type == "single":
            # Single variable flip
            for i in range(n):
                temp_assign = assignment[:]
                temp_assign[i] = 1 - temp_assign[i]
                new_h = heuristic_func(temp_assign, clauses)
                nodes_explored += 1  # Count each neighbor evaluated
                if new_h > best_h:
                    best_assign = temp_assign
                    best_h = new_h
                    improved = True
                    
        elif neigh_type == "double":
            # Double variable flip
            for i, j in itertools.combinations(range(n), 2):
                temp_assign = assignment[:]
                temp_assign[i] = 1 - temp_assign[i]
                temp_assign[j] = 1 - temp_assign[j]
                new_h = heuristic_func(temp_assign, clauses)
                nodes_explored += 1  # Count each neighbor evaluated
                if new_h > best_h:
                    best_assign = temp_assign
                    best_h = new_h
                    improved = True
                    
        elif neigh_type == "targeted":
            # Target variables from unsatisfied clauses
            unsatisfied_vars = set()
            for clause in clauses:
                clause_satisfied = False
                for lit in clause:
                    var_idx = abs(lit) - 1
                    var_val = assignment[var_idx]
                    lit_val = var_val if lit > 0 else 1 - var_val
                    if lit_val == 1:
                        clause_satisfied = True
                        break
                if not clause_satisfied:
                    for lit in clause:
                        unsatisfied_vars.add(abs(lit) - 1)
            
            for i in unsatisfied_vars:
                temp_assign = assignment[:]
                temp_assign[i] = 1 - temp_assign[i]
                new_h = heuristic_func(temp_assign, clauses)
                nodes_explored += 1  # Count each neighbor evaluated
                if new_h > best_h:
                    best_assign = temp_assign
                    best_h = new_h
                    improved = True
        
        if improved:
            assignment = best_assign
            neigh_idx = 0  # Reset to first neighborhood
        else:
            neigh_idx += 1
            if neigh_idx >= len(neighborhoods):
                break  # Tried all neighborhoods, no improvement
        
        iters += 1
    
    elapsed = time.time() - start_time
    success = evaluate(assignment, clauses) == len(clauses)
    return success, iters, elapsed, nodes_explored

def run_experiment(k: int, m_values: List[int], n_values: List[int], num_instances: int = 10, max_iters: int = 1000):
   
    algorithms = {
        "Hill-Climbing": hill_climbing,
        "Beam-3": lambda c, n, h, mi: beam_search(c, n, 3, h, mi),
        "Beam-4": lambda c, n, h, mi: beam_search(c, n, 4, h, mi),
        "VND": vnd
    }
    heuristics = {"h1": heuristic1, "h2": heuristic2}
    
    results = []
    for n in n_values:
        for m in m_values:
            for algo_name, algo_func in algorithms.items():
                for h_name, h_func in heuristics.items():
                    successes = 0
                    total_time = 0
                    total_iters = 0
                    total_nodes = 0
                    
                    for instance in range(num_instances):
                        # Generate new instance for each run
                        clauses = generate_ksat(k, m, n)
                        success, iters, t, nodes = algo_func(clauses, n, h_func, max_iters)
                        if success:
                            successes += 1
                        total_time += t
                        total_iters += iters
                        total_nodes += nodes
                    
                    avg_success = successes / num_instances * 100
                    avg_time = total_time / num_instances
                    avg_iters = total_iters / num_instances
                    avg_nodes = total_nodes / num_instances
                    
                    # Penetrance calculation
                    # Penetrance = solutions found / total nodes explored

                    penetrance = (successes / total_nodes) * 100 if total_nodes > 0 else 0
                    
                    results.append([k, m, n, algo_name, h_name, avg_success, avg_nodes, penetrance, avg_time])
    
    # Print table
    headers = ["k", "m", "n", "Algorithm", "Heuristic", "Success %", "Avg Nodes", "Penetrance %", "Avg Time (s)"]
    print(tabulate(results, headers=headers, floatfmt=".4f", tablefmt="grid"))
    
    # Print penetrance analysis
   
    print("PENETRANCE ANALYSIS")

    print("Penetrance = (Solutions Found / Total Nodes Explored) × 100")
  
    
    # Group by algorithm and heuristic for comparison
    print("Average Penetrance by Algorithm and Heuristic:")
    print("-" * 60)
    
    algo_h_stats = {}
    for row in results:
        key = (row[3], row[4])  # (algorithm, heuristic)
        if key not in algo_h_stats:
            algo_h_stats[key] = []
        algo_h_stats[key].append(row[7])  # penetrance value
    
    summary = []
    for (algo, h), pen_values in sorted(algo_h_stats.items()):
        avg_pen = sum(pen_values) / len(pen_values)
        summary.append([algo, h, avg_pen])
    
    print(tabulate(summary, headers=["Algorithm", "Heuristic", "Avg Penetrance %"], 
                   floatfmt=".4f", tablefmt="simple"))
    
    return results

# Example Usage
if __name__ == "__main__":
    # Part B Example: Generate and print a 3-SAT (k=3, m=5, n=10)
    print("Part B: Example 3-SAT Generation ===")
    example_clauses = generate_ksat(3, 5, 10)
    for i, clause in enumerate(example_clauses, 1):
        lit_strs = []
        for x in clause:
            if x < 0:
                lit_strs.append(f"~X{abs(x)}")
            else:
                lit_strs.append(f"X{x}")
        print(f"Clause {i}: {' OR '.join(lit_strs)}")
    
 
    
 
    print("\nPart C: Experiment Results :") 
    print()
    
    # Test with n=10,20 and varying m values
    n_vals = [10, 20]
    m_vals = [30, 40, 50]  # m/n ratios: 3.0, 4.0, 5.0 for n=10
    
    run_experiment(k=3, m_values=m_vals, n_values=n_vals, num_instances=5, max_iters=500)
    

 
