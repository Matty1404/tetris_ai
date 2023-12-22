import random
from tetris_ai import TetrisAI, RandomAI, Smart_AI
import numpy as np
import multiprocessing
from tetris import Tetris

class FastTetrisTrainer():
  def __init__(self, AI: TetrisAI = None):
    self.AI = AI
    self.WIDTH, self.HEIGHT = 500, 660
    self.GRID_SIZE = 30
    self.GRID_WIDTH, self.GRID_HEIGHT = 10, 22
    self.SHAPES = [
        ([[1, 1, 1, 1]], 1),
        ([[1, 1], [1, 1]], 2),
        ([[1, 1, 1], [0, 1, 0]], 3),
        ([[1, 1, 1], [1, 0, 0]], 4),
        ([[1, 1, 1], [0, 0, 1]], 5),
        ([[0, 1, 1], [1, 1, 0]], 7),
        ([[1, 1, 0], [0, 1, 1]], 8),
    ]
    self.grid = [[0] * self.GRID_WIDTH for _ in range(self.GRID_HEIGHT)]
    self.next_shape = random.choice(self.SHAPES)
    self.current_shape = random.choice(self.SHAPES)
    self.rotated = False
    self.current_x = self.GRID_WIDTH // 2
    self.current_y = 0
    self.score = 0
        
    
  def calculate_aggregate_height_and_bumpiness(self):
    # want to minimise
    prev = -1
    height = 0
    bumpiness = 0
    for i in range (0, self.GRID_WIDTH):
      broke = False
      for j in range(0, self.GRID_HEIGHT):
        if self.grid[j][i] != 0:
          height += (self.GRID_HEIGHT - j)
          broke = True
          if prev != -1:
            bumpiness += abs(prev - (self.GRID_HEIGHT - j))
          prev = self.GRID_HEIGHT - j
          break
      if not broke:
        prev = 0
    if bumpiness == 0:
      bumpiness = 0.1
    if height == 0:
      height = 0.1
    return height, bumpiness
  
  def calculate_full_lines(self):
    full_rows = []
    for row in range(self.GRID_HEIGHT):
      if all(self.grid[row]):
          full_rows.append(row)

    return len(full_rows)

  def calculate_shape_inputs(self, shape, rotated = False):
    shape = np.array(shape)
    if rotated:
      shape = shape.T
    
    padded_matrix = np.pad(shape, ((0, 4-shape.shape[0]), (0, 4-shape.shape[1])), 'constant')
    return padded_matrix.flatten().tolist()




  def play_tetris(self):
    
    final_score = 0

    game_over = False

    while not game_over:
      if self.AI is None:
        raise Exception("Need an AI to train")
      else:
        inputs = self.grid
        flattened_list = [item for sublist in inputs for item in sublist]
        
        flattened_list.extend(self.calculate_shape_inputs(self.current_shape[0], rotated=self.rotated))
        flattened_list.extend(self.calculate_shape_inputs(self.next_shape[0]))
        
        
        move = self.AI.choose_move(flattened_list)
        if move == 0: # left
          if not self.check_collision(self.current_shape[0], self.current_x - 1, self.current_y):
            self.current_x -= 1
        elif move == 1: # right
          if not self.check_collision(self.current_shape[0], self.current_x + 1, self.current_y):
            self.current_x += 1
        elif move == 2: #down
          while not self.check_collision(self.current_shape[0], self.current_x, self.current_y + 1):
            self.current_y += 1
        elif move == 3: #rotate
          rotated_shape = self.rotate_shape(self.current_shape[0])
          if not self.check_collision(rotated_shape, self.current_x, self.current_y):
            self.current_shape = (rotated_shape, self.current_shape[1])
            self.rotated = not self.rotated
        
      # Move the shape down
      if not self.check_collision(self.current_shape[0], self.current_x, self.current_y + 1):
          self.current_y += 1
      else:
          # Lock the shape in place
          for row in range(len(self.current_shape[0])):
              for col in range(len(self.current_shape[0][row])):
                  if self.current_shape[0][row][col] != 0:
                      self.grid[self.current_y + row][self.current_x + col] = self.current_shape[1]

          cleared = self.calculate_full_lines()
          self.clear_rows()
          height, bumpiness = self.calculate_aggregate_height_and_bumpiness()

          final_score += (100000 * cleared) + (1 / height) + (1 / bumpiness) 

          # Generate a new random shape
          self.current_shape = self.next_shape
          self.next_shape = random.choice(self.SHAPES)
          self.current_x = self.GRID_WIDTH // 2
          self.current_y = 0

          # Check if the game is over
          if self.check_collision(self.current_shape[0], self.current_x, self.current_y):
              game_over = True



      
      
    # Game over screen
    return final_score

  def check_collision(self, shape, x, y):
    for row in range(len(shape)):
      for col in range(len(shape[row])):
        if shape[row][col] != 0:
          if x + col < 0 or x + col >= self.GRID_WIDTH or y + row >= self.GRID_HEIGHT or self.grid[y + row][x + col] != 0:
            return True
    return False

  def rotate_shape(self, shape):
    return [[shape[y][x] for y in range(len(shape))] for x in range(len(shape[0]) - 1, -1, -1)]

  def clear_rows(self):
    full_rows = []
    for row in range(self.GRID_HEIGHT):
      if all(self.grid[row]):
        full_rows.append(row)


      # number of complete lines is len(full_rows)

      for row in full_rows:
          self.grid.pop(row)
          self.grid.insert(0, [0] * self.GRID_WIDTH)
          self.score += 100 * len(full_rows)



# Tetris().play_tetris()
# Tetris(Smart_AI()).play_tetris()
# Tetris(RandomAI()).play_tetris()

def cross_and_mutate(num_tournaments, top_n):
    
  tournament_size = 6 #TODO change
  new_ais = []
  print("HOLDING TOURNAMENT")
  def tournament_selection(population, tournament_size):
    # Randomly select tournament_size genomes from the population
    tournament_candidates = random.sample(population, tournament_size)

    # Evaluate the fitness of each candidate and select the fittest
    winner = max(tournament_candidates, key=lambda x: x[0])

    return winner

  # Perform a series of tournaments to choose two genomes for crossover
  for _ in range(num_tournaments):
      # Perform tournament selection to choose the first parent
      parent1 = tournament_selection(top_n, tournament_size)

      # Perform tournament selection to choose the second parent (ensure it's different from the first)
      parent2 = tournament_selection(top_n, tournament_size)
      while parent2 == parent1:
          parent2 = tournament_selection(top_n, tournament_size)

      # Now you have two parents (parent1 and parent2) for crossover
      
      new_ais.append(Smart_AI(parent1[1].genome, parent2[1].genome))

      
  
  
  
  
  assert (num_tournaments == len(new_ais))
  return new_ais

def run_tetris_instance(ai, result_queue, id):
  print("running instance")
  score = FastTetrisTrainer(ai).play_tetris()
  print(score)
  result_queue.put((score, id))

if __name__ == "__main__":
  ai_instances = [Smart_AI() for _ in range(90)] 
  best_10 = [Smart_AI() for _ in range(10)] 
  all_ai = best_10 + ai_instances

  for i in range(1000): 
    processes = []
    
    
    result_queue = multiprocessing.Queue()
    for id, ai in enumerate(best_10):
      process = multiprocessing.Process(target=run_tetris_instance, args=(ai,result_queue, id))
      processes.append((process))
      process.start()
    for id, ai in enumerate(ai_instances):
      print(ai)
      process = multiprocessing.Process(target=run_tetris_instance, args=(ai,result_queue, id + 10))
      processes.append((process))
      process.start()

    for process in processes:
      process.join()
    results = []
    while not result_queue.empty():
      result, id = result_queue.get()
      print(result, id)
      results.append((result, all_ai[id]))
    results.sort(reverse=True)
    print(results)
    new_population = results[:10] 
    ai_instances = cross_and_mutate(90, new_population) 
    best_10 = list(map(lambda x: x[1], new_population))
  
  
  for index, ai in enumerate(best_10):
    with open("genomes.txt", "w") as f:
      f.write("---------------- seperator ----------------------\n\n")
      f.write(ai.genome)

  Tetris(best_10[0]).play_tetris()
  
  
  
  
  # revert values
  # do cross section and mutation
  # change tick rate so faster