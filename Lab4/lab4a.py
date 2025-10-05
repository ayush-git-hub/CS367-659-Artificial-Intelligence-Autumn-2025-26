import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random
import math
from copy import deepcopy

def load_scrambled_image(filename):
    """Load the scrambled image from .mat file"""
    try:
        # Try loading as MATLAB file2
        data = loadmat(filename)
        # Extract the image array (common keys: 'img', 'image', or first variable)
        for key in data.keys():
            if not key.startswith('__'):
                image = data[key]
                break
    except:
        # Fallback to ASCII format
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        data_lines = []
        start_reading = False
        
        for line in lines:
            if not start_reading:
                if line.strip() == "512 512":
                    start_reading = True
                continue
            
            if line.strip() and not line.startswith('#'):
                data_lines.append(line)
        
        data = np.loadtxt(data_lines, dtype=np.uint8)
        image = data.reshape((512, 512))
    
    return image

class JigsawPuzzle:
    """Jigsaw Puzzle Solver using Simulated Annealing"""
    
    def __init__(self, scrambled_image, tile_size=64):
        """
        Initialize the puzzle
        
        Parameters:
        -----------
        scrambled_image : numpy array
            The scrambled image
        tile_size : int
            Size of each tile (assuming square tiles)
        """
        self.scrambled_image = scrambled_image.astype(float)
        self.tile_size = tile_size
        self.img_height, self.img_width = scrambled_image.shape
        
        # Calculate grid dimensions
        self.grid_rows = self.img_height // tile_size
        self.grid_cols = self.img_width // tile_size
        self.num_tiles = self.grid_rows * self.grid_cols
        
        print(f"Image size: {self.img_height}x{self.img_width}")
        print(f"Tile size: {tile_size}x{tile_size}")
        print(f"Grid: {self.grid_rows}x{self.grid_cols} ({self.num_tiles} tiles)")
        
        # Extract tiles from scrambled image
        self.tiles = self._extract_tiles()
        
        # Initialize with identity permutation (scrambled state)
        self.current_permutation = list(range(self.num_tiles))
        self.best_permutation = self.current_permutation.copy()
        self.best_energy = float('inf')
        
    def _extract_tiles(self):
        """Extract individual tiles from the scrambled image"""
        tiles = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                tile = self.scrambled_image[
                    row * self.tile_size:(row + 1) * self.tile_size,
                    col * self.tile_size:(col + 1) * self.tile_size
                ]
                tiles.append(tile)
        return tiles
    
    def _reconstruct_image(self, permutation):
        """Reconstruct image from tiles using given permutation"""
        reconstructed = np.zeros_like(self.scrambled_image)
        
        for idx, tile_idx in enumerate(permutation):
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            
            reconstructed[
                row * self.tile_size:(row + 1) * self.tile_size,
                col * self.tile_size:(col + 1) * self.tile_size
            ] = self.tiles[tile_idx]
        
        return reconstructed
    
    def _calculate_boundary_difference(self, tile1, tile2, direction='horizontal'):
        """
        Calculate the difference at the boundary between two tiles
        
        Parameters:
        -----------
        tile1, tile2 : numpy arrays
            The two tiles to compare
        direction : str
            'horizontal' if tile2 is to the right of tile1
            'vertical' if tile2 is below tile1
        """
        if direction == 'horizontal':
            # Compare right edge of tile1 with left edge of tile2
            edge1 = tile1[:, -1]
            edge2 = tile2[:, 0]
        else:  # vertical
            # Compare bottom edge of tile1 with top edge of tile2
            edge1 = tile1[-1, :]
            edge2 = tile2[0, :]
        
        # Mean squared difference
        return np.sum((edge1 - edge2) ** 2)
    
    def calculate_energy(self, permutation):
        """
        Calculate the energy (cost) of a given permutation
        Lower energy means better fit between adjacent tiles
        """
        energy = 0.0
        
        for idx, tile_idx in enumerate(permutation):
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            
            # Check right neighbor
            if col < self.grid_cols - 1:
                right_idx = idx + 1
                right_tile_idx = permutation[right_idx]
                energy += self._calculate_boundary_difference(
                    self.tiles[tile_idx], 
                    self.tiles[right_tile_idx], 
                    'horizontal'
                )
            
            # Check bottom neighbor
            if row < self.grid_rows - 1:
                bottom_idx = idx + self.grid_cols
                bottom_tile_idx = permutation[bottom_idx]
                energy += self._calculate_boundary_difference(
                    self.tiles[tile_idx], 
                    self.tiles[bottom_tile_idx], 
                    'vertical'
                )
        
        return energy
    
    def get_neighbor_permutation(self, permutation, temperature=None, initial_temp=None):
        """Generate a neighbor state by swapping tiles with adaptive strategies"""
        neighbor = permutation.copy()
        
        # Adaptive strategy selection based on temperature
        if temperature and initial_temp:
            temp_ratio = temperature / initial_temp
            # Early phase: more random exploration
            if temp_ratio > 0.7:
                strategy = random.choice(['swap_random', 'swap_random', 'swap_row', 'swap_col'])
            # Middle phase: balanced
            elif temp_ratio > 0.3:
                strategy = random.choice(['swap_random', 'swap_adjacent', 'swap_row', 'swap_col', 'swap_block'])
            # Late phase: more local moves
            else:
                strategy = random.choice(['swap_adjacent', 'swap_adjacent', 'swap_random', 'swap_block'])
        else:
            strategy = random.choice(['swap_random', 'swap_adjacent', 'swap_row', 'swap_col', 'swap_block'])
        
        if strategy == 'swap_random':
            # Swap two random tiles
            i, j = random.sample(range(self.num_tiles), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
        elif strategy == 'swap_adjacent':
            # Swap adjacent tiles
            idx = random.randint(0, self.num_tiles - 1)
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            
            # Choose a valid adjacent position
            adjacent = []
            if col < self.grid_cols - 1:  # right
                adjacent.append(idx + 1)
            if col > 0:  # left
                adjacent.append(idx - 1)
            if row < self.grid_rows - 1:  # bottom
                adjacent.append(idx + self.grid_cols)
            if row > 0:  # top
                adjacent.append(idx - self.grid_cols)
            
            if adjacent:
                swap_idx = random.choice(adjacent)
                neighbor[idx], neighbor[swap_idx] = neighbor[swap_idx], neighbor[idx]
                    
        elif strategy == 'swap_row':
            # Swap two tiles in the same row
            row = random.randint(0, self.grid_rows - 1)
            col1, col2 = random.sample(range(self.grid_cols), 2)
            idx1 = row * self.grid_cols + col1
            idx2 = row * self.grid_cols + col2
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            
        elif strategy == 'swap_col':
            # Swap two tiles in the same column
            col = random.randint(0, self.grid_cols - 1)
            row1, row2 = random.sample(range(self.grid_rows), 2)
            idx1 = row1 * self.grid_cols + col
            idx2 = row2 * self.grid_cols + col
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            
        else:  # swap_block
            # Swap a 2x2 block with another 2x2 block
            if self.grid_rows >= 2 and self.grid_cols >= 2:
                row1 = random.randint(0, self.grid_rows - 2)
                col1 = random.randint(0, self.grid_cols - 2)
                row2 = random.randint(0, self.grid_rows - 2)
                col2 = random.randint(0, self.grid_cols - 2)
                
                for dr in range(2):
                    for dc in range(2):
                        idx1 = (row1 + dr) * self.grid_cols + (col1 + dc)
                        idx2 = (row2 + dr) * self.grid_cols + (col2 + dc)
                        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            else:
                # Fallback to random swap
                i, j = random.sample(range(self.num_tiles), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        return neighbor
    
    def simulated_annealing(self, initial_temp=100000, cooling_rate=0.9995, 
                          min_temp=0.01, max_iterations=100000, restarts=5):
        """
        Solve the puzzle using simulated annealing with multiple restarts
        
        Parameters:
        -----------
        initial_temp : float
            Starting temperature
        cooling_rate : float
            Rate at which temperature decreases (0 < cooling_rate < 1)
        min_temp : float
            Minimum temperature (stopping criterion)
        max_iterations : int
            Maximum number of iterations per restart
        restarts : int
            Number of random restarts
        """
        global_best_permutation = None
        global_best_energy = float('inf')
        
        print(f"\nStarting Simulated Annealing with {restarts} restarts...")
        
        for restart in range(restarts):
            print(f"\n{'='*60}")
            print(f"RESTART {restart + 1}/{restarts}")
            print(f"{'='*60}")
            
            # Start with a random permutation
            self.current_permutation = list(range(self.num_tiles))
            random.shuffle(self.current_permutation)
            
            current_energy = self.calculate_energy(self.current_permutation)
            self.best_permutation = self.current_permutation.copy()
            self.best_energy = current_energy
            
            temperature = initial_temp
            iteration = 0
            stagnant_count = 0
            
            energy_history = []
            temp_history = []
            
            print(f"Initial energy: {current_energy:.2f}")
            
            while temperature > min_temp and iteration < max_iterations:
                # Generate neighbor with adaptive strategy
                neighbor_perm = self.get_neighbor_permutation(self.current_permutation, temperature, initial_temp)
                neighbor_energy = self.calculate_energy(neighbor_perm)
                
                # Calculate energy difference
                delta_energy = neighbor_energy - current_energy
                
                # Acceptance criterion
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
                    self.current_permutation = neighbor_perm
                    current_energy = neighbor_energy
                    stagnant_count = 0
                    
                    # Update best solution
                    if current_energy < self.best_energy:
                        self.best_permutation = self.current_permutation.copy()
                        self.best_energy = current_energy
                        print(f"Iteration {iteration}: New best energy = {self.best_energy:.2f}, Temp = {temperature:.2f}")
                else:
                    stagnant_count += 1
                
                # Adaptive reheating if stuck
                if stagnant_count > 2000:
                    temperature = min(temperature * 3, initial_temp * 0.5)
                    stagnant_count = 0
                    print(f"Reheating at iteration {iteration}, Temp = {temperature:.2f}")
                
                # Cool down
                temperature *= cooling_rate
                iteration += 1
                
                # Record history
                if iteration % 100 == 0:
                    energy_history.append(current_energy)
                    temp_history.append(temperature)
                
                # Progress update
                if iteration % 5000 == 0:
                    print(f"Iteration {iteration}/{max_iterations}, Energy: {current_energy:.2f}, Temp: {temperature:.4f}")
                
                # Early stopping if solution is excellent
                if self.best_energy < 50000:  # Lower threshold for better solution
                    print(f"Found excellent solution at iteration {iteration}!")
                    break
            
            print(f"\nRestart {restart + 1} complete: Best energy = {self.best_energy:.2f}")
            
            # Update global best
            if self.best_energy < global_best_energy:
                global_best_energy = self.best_energy
                global_best_permutation = self.best_permutation.copy()
                print(f"*** New global best energy: {global_best_energy:.2f} ***")
        
        # Set the global best as final result
        self.best_permutation = global_best_permutation
        self.best_energy = global_best_energy
        
        print(f"\n{'='*60}")
        print(f"Optimization complete!")
        print(f"Final best energy: {self.best_energy:.2f}")
        print(f"{'='*60}")
        
        return energy_history, temp_history
    
    def visualize_results(self, energy_history=None, temp_history=None):
        """Visualize the scrambled, reconstructed, and energy history"""
        fig = plt.figure(figsize=(15, 5))
        
        # Scrambled image
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(self.scrambled_image, cmap='gray')
        ax1.set_title('Scrambled Image')
        ax1.axis('off')
        
        # Reconstructed image
        reconstructed = self._reconstruct_image(self.best_permutation)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(reconstructed, cmap='gray')
        ax2.set_title(f'Reconstructed Image\nEnergy: {self.best_energy:.2f}')
        ax2.axis('off')
        
        # Energy history
        if energy_history:
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.plot(energy_history, 'b-', linewidth=2)
            ax3.set_xlabel('Iteration (x100)')
            ax3.set_ylabel('Energy')
            ax3.set_title('Energy vs Iteration')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('puzzle_solution.png', dpi=150, bbox_inches='tight')
        print("\nResults saved to 'puzzle_solution.png'")
        plt.show()
        
        return reconstructed
    
    def save_solution(self, filename='solved_image.png'):
        """Save the reconstructed image"""
        reconstructed = self._reconstruct_image(self.best_permutation)
        plt.figure(figsize=(8, 8))
        plt.imshow(reconstructed, cmap='gray')
        plt.axis('off')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Solved image saved to '{filename}'")
        plt.close()


def main():
    """Main function to run the puzzle solver"""
    print("="*60)
    print("JIGSAW PUZZLE SOLVER USING SIMULATED ANNEALING")
    print("="*60)
    
    # Load scrambled image
    print("\nLoading scrambled image...")
    scrambled_img = load_scrambled_image('scrambled_lena.mat')
    
    # Create puzzle solver
    # For 512x512 image with 4x4 grid: tile_size = 128
    tile_size = 128  # 512/4 = 128 pixels per tile
    puzzle = JigsawPuzzle(scrambled_img, tile_size=tile_size)
    
    # Solve using simulated annealing
    print("\n" + "="*60)
    energy_history, temp_history = puzzle.simulated_annealing(
        initial_temp=150000,
        cooling_rate=0.9996,
        min_temp=0.01,
        max_iterations=150000,
        restarts=15
    )
    
    # Visualize results
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    reconstructed = puzzle.visualize_results(energy_history, temp_history)
    puzzle.save_solution('solved_lena.png')
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == "__main__":
    main()