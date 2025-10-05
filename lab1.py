# Lab 1 
# Problem Statement
# In the rabbit leap problem, three east-bound rabbits stand in a line blocked by three west-bound rabbits. They are crossing a stream with stones placed in the east west direction in a line. There is one empty stone between them. The rabbits can only move forward one or two steps. They can jump over one rabbit if the need arises, but not more than that. Are they smart enough to cross each other without having to step into the water? 

from collections import deque
from typing import List, Optional, Tuple, Set
from abc import ABC, abstractmethod
import time
import math

class State:
    """Represents a state in the rabbit leap problem."""
    def __init__(self, configuration: str, parent: Optional['State'] = None, 
                 action: str = "Initial State", depth: int = 0):
        self.configuration = configuration
        self.parent = parent
        self.action = action
        self.depth = depth
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.configuration == other.configuration
    
    def __hash__(self):
        return hash(self.configuration)
    
    def __str__(self):
        return self.configuration
    
    def __repr__(self):
        return f"State('{self.configuration}', depth={self.depth})"
    
    def get_path(self) -> List[Tuple['State', str]]:
        """Returns the path from initial state to this state."""
        path = []
        current = self
        while current is not None:
            path.append((current, current.action))
            current = current.parent
        return list(reversed(path))

class RabbitLeapProblem:
    """Models the rabbit leap problem as a state space search problem."""
    def __init__(self, initial_state: str = "EEE_WWW", goal_state: str = "WWW_EEE"):
        self.initial_state = State(initial_state)
        self.goal_state_config = goal_state
        self.total_positions = len(initial_state)
    
    def is_goal(self, state: State) -> bool:
        """Check if the given state is the goal state."""
        return state.configuration == self.goal_state_config
    
    def get_empty_position(self, state: State) -> int:
        """Find the position of the empty stone."""
        return state.configuration.index('_')
    
    def get_successors(self, state: State) -> List[State]:
        """Generate all valid successor states from the current state."""
        successors = []
        config = list(state.configuration)
        empty_pos = self.get_empty_position(state)
        
        # East rabbit moves right 1 step
        if empty_pos > 0 and config[empty_pos - 1] == 'E':
            new_config = config.copy()
            new_config[empty_pos], new_config[empty_pos - 1] = new_config[empty_pos - 1], new_config[empty_pos]
            action = f"E at position {empty_pos - 1} moves right 1"
            successors.append(State(''.join(new_config), state, action, state.depth + 1))
        
        # East rabbit jumps right 2 steps (over a West rabbit)
        if empty_pos > 1 and config[empty_pos - 2] == 'E' and config[empty_pos - 1] == 'W':
            new_config = config.copy()
            new_config[empty_pos], new_config[empty_pos - 2] = new_config[empty_pos - 2], new_config[empty_pos]
            action = f"E at position {empty_pos - 2} jumps right 2"
            successors.append(State(''.join(new_config), state, action, state.depth + 1))
        
        # West rabbit moves left 1 step
        if empty_pos < self.total_positions - 1 and config[empty_pos + 1] == 'W':
            new_config = config.copy()
            new_config[empty_pos], new_config[empty_pos + 1] = new_config[empty_pos + 1], new_config[empty_pos]
            action = f"W at position {empty_pos + 1} moves left 1"
            successors.append(State(''.join(new_config), state, action, state.depth + 1))
        
        # West rabbit jumps left 2 steps (over an East rabbit)
        if empty_pos < self.total_positions - 2 and config[empty_pos + 2] == 'W' and config[empty_pos + 1] == 'E':
            new_config = config.copy()
            new_config[empty_pos], new_config[empty_pos + 2] = new_config[empty_pos + 2], new_config[empty_pos]
            action = f"W at position {empty_pos + 2} jumps left 2"
            successors.append(State(''.join(new_config), state, action, state.depth + 1))
        
        return successors

class SearchAlgorithm(ABC):
    """Abstract base class for search algorithms."""
    def __init__(self, problem: RabbitLeapProblem):
        self.problem = problem
        self.nodes_explored = 0
        self.max_frontier_size = 0
        self.execution_time = 0.0
    
    @abstractmethod
    def search(self) -> Optional[State]:
        """Execute the search algorithm."""
        pass
    
    def get_statistics(self) -> dict:
        """Return search statistics."""
        return {
            'nodes_explored': self.nodes_explored,
            'max_frontier_size': self.max_frontier_size,
            'execution_time': self.execution_time
        }
    
    def print_solution(self, goal_state: Optional[State]):
        """Print the solution path and statistics."""
        if goal_state is None:
            print("No solution found!")
            return
        
        path = goal_state.get_path()
        print(f"\n{'='*70}")
        print(f"Solution found with {len(path) - 1} moves!")
        print(f"{'='*70}\n")
        
        for i, (state, action) in enumerate(path):
            print(f"Step {i}: {state.configuration}")
            if action != "Initial State":
                print(f"       Action: {action}")
            print()
        
        print(f"{'='*70}")
        print(f"Total moves: {len(path) - 1}")
        print(f"Nodes explored: {self.nodes_explored}")
        print(f"Max frontier size: {self.max_frontier_size}")
        print(f"Execution time: {self.execution_time:.6f} seconds")
        print(f"{'='*70}\n")

class BreadthFirstSearch(SearchAlgorithm):
    """Breadth-First Search implementation - guarantees optimal solution."""
    def search(self) -> Optional[State]:
        start_time = time.time()
        frontier = deque([self.problem.initial_state])
        explored: Set[str] = set()
        visited: Set[str] = {self.problem.initial_state.configuration}
        
        self.nodes_explored = 0
        self.max_frontier_size = 1
        
        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            current_state = frontier.popleft()
            explored.add(current_state.configuration)
            self.nodes_explored += 1
            
            if self.problem.is_goal(current_state):
                self.execution_time = time.time() - start_time
                return current_state
            
            for successor in self.problem.get_successors(current_state):
                if successor.configuration not in visited:
                    visited.add(successor.configuration)
                    frontier.append(successor)
        
        self.execution_time = time.time() - start_time
        return None

class DepthFirstSearch(SearchAlgorithm):
    """Depth-First Search implementation - may not find optimal solution."""
    def search(self) -> Optional[State]:
        start_time = time.time()
        frontier = [self.problem.initial_state]
        explored: Set[str] = set()
        
        self.nodes_explored = 0
        self.max_frontier_size = 1
        
        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            current_state = frontier.pop()
            
            if current_state.configuration in explored:
                continue
            
            explored.add(current_state.configuration)
            self.nodes_explored += 1
            
            if self.problem.is_goal(current_state):
                self.execution_time = time.time() - start_time
                return current_state
            
            successors = self.problem.get_successors(current_state)
            for successor in reversed(successors):
                if successor.configuration not in explored:
                    frontier.append(successor)
        
        self.execution_time = time.time() - start_time
        return None

class StateSpaceAnalyzer:
    """Analyzes the state space of the problem."""
    def __init__(self, problem: RabbitLeapProblem):
        self.problem = problem
    
    def calculate_theoretical_states(self) -> int:
        """Calculate theoretical maximum number of states."""
        n = self.problem.total_positions
        config = self.problem.initial_state.configuration
        num_e = config.count('E')
        num_w = config.count('W')
        num_empty = config.count('_')
        
        # Permutations with repetition: n! / (n_E! * n_W! * n_empty!)
        numerator = math.factorial(n)
        denominator = math.factorial(num_e) * math.factorial(num_w) * math.factorial(num_empty)
        return numerator // denominator
    
    def explore_reachable_states(self) -> Set[str]:
        """Find all states reachable from the initial state."""
        frontier = deque([self.problem.initial_state])
        visited = {self.problem.initial_state.configuration}
        
        while frontier:
            current = frontier.popleft()
            for successor in self.problem.get_successors(current):
                if successor.configuration not in visited:
                    visited.add(successor.configuration)
                    frontier.append(successor)
        
        return visited
    
    def analyze(self):
        """Perform and print state space analysis."""
        print(f"\n{'='*70}")
        print("STATE SPACE ANALYSIS")
        print(f"{'='*70}\n")
        
        theoretical = self.calculate_theoretical_states()
        print(f"Theoretical maximum states: {theoretical}")
        print(f"  (All possible arrangements of 3 E's, 3 W's, and 1 empty space)")
        
        reachable = self.explore_reachable_states()
        print(f"\nActual reachable states: {len(reachable)}")
        print(f"  (States reachable following movement rules)")
        
        percentage = (len(reachable) / theoretical) * 100
        print(f"\nPercentage reachable: {percentage:.2f}%")
        print(f"  (Not all arrangements can be reached due to movement constraints)")
        
        print(f"\n{'='*70}\n")

def main():
    """Main function to run the rabbit leap problem solver."""
    print("\n" + "="*70)
    print(" " * 20 + "RABBIT LEAP PROBLEM SOLVER")
    print("="*70 + "\n")
    
    # Initialize problem
    problem = RabbitLeapProblem()
    print(f"Initial State: {problem.initial_state}")
    print(f"Goal State:    {problem.goal_state_config}")
    print("\nLegend: E = East-bound rabbit, W = West-bound rabbit, _ = empty stone\n")
    
    # Analyze state space
    analyzer = StateSpaceAnalyzer(problem)
    analyzer.analyze()
    
    # Solve with BFS
    print("\n" + "="*70)
    print(" " * 25 + "BFS SOLUTION")
    print("="*70)
    bfs = BreadthFirstSearch(problem)
    bfs_solution = bfs.search()
    bfs.print_solution(bfs_solution)
    
    # Solve with DFS
    print("\n" + "="*70)
    print(" " * 25 + "DFS SOLUTION")
    print("="*70)
    dfs = DepthFirstSearch(problem)
    dfs_solution = dfs.search()
    dfs.print_solution(dfs_solution)
    
    # Compare algorithms
    print("\n" + "="*70)
    print(" " * 20 + "ALGORITHM COMPARISON")
    print("="*70 + "\n")
    
    if bfs_solution and dfs_solution:
        bfs_path_length = len(bfs_solution.get_path()) - 1
        dfs_path_length = len(dfs_solution.get_path()) - 1
        
        print(f"{'Metric':<30} {'BFS':<20} {'DFS':<20}")
        print("-" * 70)
        print(f"{'Solution Length (moves)':<30} {bfs_path_length:<20} {dfs_path_length:<20}")
        print(f"{'Nodes Explored':<30} {bfs.nodes_explored:<20} {dfs.nodes_explored:<20}")
        print(f"{'Max Frontier Size':<30} {bfs.max_frontier_size:<20} {dfs.max_frontier_size:<20}")
        print(f"{'Execution Time (seconds)':<30} {bfs.execution_time:<20.6f} {dfs.execution_time:<20.6f}")
        print(f"{'Optimal Solution':<30} {'Yes':<20} {'No' if dfs_path_length > bfs_path_length else 'Yes':<20}")
        print(f"{'Space Complexity':<30} {'O(b^d)':<20} {'O(bd)':<20}")
        print(f"{'Time Complexity':<30} {'O(b^d)':<20} {'O(b^m)':<20}")
        
        print(f"\n{'ANALYSIS:':<30}")
        print(f"  - BFS guarantees the optimal (shortest) solution")
        print(f"  - BFS explored {bfs.nodes_explored} nodes vs DFS's {dfs.nodes_explored} nodes")
        print(f"  - BFS uses more memory (frontier size: {bfs.max_frontier_size} vs {dfs.max_frontier_size})")
        print(f"  - DFS is {'not ' if dfs_path_length > bfs_path_length else ''}optimal for this problem")
        
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()