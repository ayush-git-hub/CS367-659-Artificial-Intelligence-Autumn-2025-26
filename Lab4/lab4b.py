import random
import math

# Raag Bhairav notes: Sa Re(komal) Ga Ma Pa Dha(komal) Ni Sa
notes = ['S', 'r', 'G', 'M', 'P', 'd', 'N', 'S2']
vadi = 'r'  # Most important note
samvadi = 'd'  # Second most important

def calculate_fitness(melody):
    score = 0
    
    # 1. Emphasis on vadi and samvadi (but not too much)
    vadi_count = melody.count(vadi)
    samvadi_count = melody.count(samvadi)
    score += min(vadi_count * 3, 15)  # Cap the reward
    score += min(samvadi_count * 2, 10)
    
    # 2. Start and end properly
    if melody[0] == 'S': score += 20
    if melody[-1] in ['S', 'S2']: score += 20
    
    # 3. Smooth melodic movement
    for i in range(len(melody) - 1):
        jump = abs(notes.index(melody[i+1]) - notes.index(melody[i]))
        if jump == 1: score += 5  # Step movement is best
        elif jump == 2: score += 3
        elif jump == 3: score += 1
        elif jump > 4: score -= 10  # Penalize big jumps
    
    # 4. Consonant intervals
    consonant_pairs = [('S','P'),('P','S'),('S','M'),('M','S'),('r','d'),('d','r'),
                       ('S','G'),('G','S'),('P','N'),('N','P'),('G','M'),('M','G')]
    for i in range(len(melody) - 1):
        if (melody[i], melody[i+1]) in consonant_pairs:
            score += 4
    
    # 5. Diversity - penalize too much repetition
    unique_notes = len(set(melody))
    score += unique_notes * 3
    
    # 6. Penalize consecutive same notes heavily
    for i in range(len(melody) - 1):
        if melody[i] == melody[i+1]:
            score -= 8
    
    # 7. Reward characteristic phrases
    melody_str = ''.join(melody)
    if 'SrG' in melody_str: score += 10
    if 'rGM' in melody_str: score += 10
    if 'PdN' in melody_str: score += 8
    if 'MGr' in melody_str: score += 8
    
    # 8. Penalize if melody is too monotonous
    if vadi_count > len(melody) * 0.5: score -= 30
    if any(melody.count(n) > len(melody) * 0.4 for n in notes): score -= 25
    
    return score

def get_neighbor(melody):
    neighbor = melody.copy()
    # Try different mutation strategies
    strategy = random.randint(1, 3)
    
    if strategy == 1:
        # Change one random note
        idx = random.randint(0, len(neighbor) - 1)
        neighbor[idx] = random.choice(notes)
    elif strategy == 2:
        # Swap two adjacent notes
        idx = random.randint(0, len(neighbor) - 2)
        neighbor[idx], neighbor[idx+1] = neighbor[idx+1], neighbor[idx]
    else:
        # Change two random notes
        idx1 = random.randint(0, len(neighbor) - 1)
        idx2 = random.randint(0, len(neighbor) - 1)
        neighbor[idx1] = random.choice(notes)
        neighbor[idx2] = random.choice(notes)
    
    return neighbor

def simulated_annealing():
    melody_length = 16
    
    # Better initialization - start with Sa and end with Sa
    current = [random.choice(notes) for _ in range(melody_length)]
    current[0] = 'S'
    current[-1] = 'S'
    
    current_fitness = calculate_fitness(current)
    best = current.copy()
    best_fitness = current_fitness
    temperature = 150
    
    for iteration in range(5000):
        neighbor = get_neighbor(current)
        # Keep Sa at start and end
        neighbor[0] = 'S'
        neighbor[-1] = 'S'
        
        neighbor_fitness = calculate_fitness(neighbor)
        delta = neighbor_fitness - current_fitness
        
        if delta > 0 or random.random() < math.exp(delta / temperature):
            current = neighbor
            current_fitness = neighbor_fitness
            if current_fitness > best_fitness:
                best = current
                best_fitness = current_fitness
        
        temperature *= 0.995
    
    return best, best_fitness

# Generate melody
print("Generating Raag Bhairav melody using Simulated Annealing...\n")
melody, fitness = simulated_annealing()

note_names = {
    'S': 'Sa', 'r': 'Re(k)', 'G': 'Ga', 'M': 'Ma',
    'P': 'Pa', 'd': 'Dha(k)', 'N': 'Ni', 'S2': 'Sa(upper)'
}

print("Generated Melody:")
print("Notes:     ", ' - '.join(melody))
print("Swaras:    ", ' - '.join([note_names[n] for n in melody]))
print(f"\nFitness Score: {fitness}")
print(f"Unique notes used: {len(set(melody))}")
print(f"Vadi (Re komal) count: {melody.count('r')}")
print(f"Samvadi (Dha komal) count: {melody.count('d')}")