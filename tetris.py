import pygame
import random
from tetris_ai import TetrisAI, RandomAI, Smart_AI
import numpy as np

class Tetris():
  def __init__(self, AI: TetrisAI = None):
    self.AI = AI
    self.COLOURS = {
            1: pygame.Color(255, 0, 0, a=255), 
            2: pygame.Color(0, 255, 0, a=255), 
            3: pygame.Color(0, 0, 255, a=255),
            4: pygame.Color(255, 255, 0, a=255), 
            5: pygame.Color(255, 0, 255, a=255), 
            6: pygame.Color(0, 255, 255, a=255),
            7: pygame.Color(160, 32, 240, a=255),
            8: pygame.Color(255, 165, 0, a=255)
          }
    self.WIDTH, self.HEIGHT = 1000, 660
    self.GRID_SIZE = 30
    self.GRID_WIDTH, self.GRID_HEIGHT = 10, 22
    self.WHITE = (255, 255, 255)
    self.BLACK = (0, 0, 0)
    self.SHAPES = [
        ([[1, 1, 1, 1]], 1),
        ([[1, 1], [1, 1]], 2),
        ([[1, 1, 1], [0, 1, 0]], 3),
        ([[1, 1, 1], [1, 0, 0]], 4),
        ([[1, 1, 1], [0, 0, 1]], 5),
        ([[1, 1, 1], [1, 1, 1]], 6),
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
    
    pygame.init()
    self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
    
    
  def calculate_aggregate_height_and_bumpiness_and_holes(self):
    # want to minimise
    prev = -1
    height = 0
    bumpiness = 0
    holes = 0
    for i in range (0, self.GRID_WIDTH):
      broke = False
      for j in range(0, self.GRID_HEIGHT):
        
        if self.grid[j][i] == 0 and  j - 1 >= 0 and self.grid[j - 1][i] != 0:
          holes += 1
        
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
    return height, bumpiness, holes
  
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

  def calculate_holes(self):
    holes = 0
    for i in range (self.GRID_WIDTH):
      for j in range(1, self.GRID_HEIGHT):
        if self.grid[j][i] == 0 and self.grid[j - 1][i] != 0:
          holes += 1
    return holes


  def play_tetris(self):
    
    pygame.display.set_caption("Tetris")

    # Initialize clock
    clock = pygame.time.Clock()

    # Define fonts
    font = pygame.font.Font(None, 36)
    
    # Main game loop
    game_over = False

    while not game_over:
      if self.AI is None:
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            game_over = True
          if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
              if not self.check_collision(self.current_shape[0], self.current_x - 1, self.current_y):
                self.current_x -= 1
            elif event.key == pygame.K_RIGHT:
              if not self.check_collision(self.current_shape[0], self.current_x + 1, self.current_y):
                self.current_x += 1
            elif event.key == pygame.K_DOWN:
              while not self.check_collision(self.current_shape[0], self.current_x, self.current_y + 1):
                self.current_y += 1
            elif event.key == pygame.K_UP:
              rotated_shape = self.rotate_shape(self.current_shape[0])
              if not self.check_collision(rotated_shape, self.current_x, self.current_y):
                self.current_shape = (rotated_shape, self.current_shape[1])
                self.rotated = not self.rotated
      else:
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            game_over = True
            pygame.quit()
            exit()
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

          self.clear_rows()
          holes = self.calculate_holes()
          print(holes)

          # Generate a new random shape
          self.current_shape = self.next_shape
          self.next_shape = random.choice(self.SHAPES)
          self.current_x = self.GRID_WIDTH // 2
          self.current_y = 0

          # Check if the game is over
          if self.check_collision(self.current_shape[0], self.current_x, self.current_y):
              game_over = True

      # Clear the screen
      self.screen.fill(self.BLACK)

      # Draw the grid and current shape
      self.draw_grid()
      self.draw_shape(self.current_shape, self.current_x, self.current_y)

      # Draw the score
      score_text = font.render(f"Score: {self.score}", True, self.WHITE)
      self.screen.blit(score_text, (10, 10))

      # Update the display
      pygame.display.flip()

      # Limit the frame rate
      clock.tick(5)
    # Game over screen
    game_over_text = font.render("Game Over", True, self.WHITE)
    self.screen.blit(game_over_text, (self.WIDTH // 2 - game_over_text.get_width() // 2, self.HEIGHT // 2 - game_over_text.get_height() // 2))
    pygame.display.flip()

    # Wait for a key press to exit
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                pygame.quit()
                exit()

    
    
    
    

  # Functions
  def draw_grid(self):
    for row in range(self.GRID_HEIGHT):
      pygame.draw.line(self.screen, self.WHITE, (0, row * self.GRID_SIZE), (self.GRID_WIDTH * self.GRID_SIZE, row * self.GRID_SIZE), 1)
    for col in range(self.GRID_WIDTH + 1):
      pygame.draw.line(self.screen, self.WHITE, (col * self.GRID_SIZE, 0), (col * self.GRID_SIZE, self.HEIGHT), 1)
      
    for row in range(2,7):
      pygame.draw.line(self.screen, self.WHITE, ((self.GRID_WIDTH + 1) * self.GRID_SIZE, row * self.GRID_SIZE), ((self.GRID_WIDTH + 1) * self.GRID_SIZE + 4 * self.GRID_SIZE, row * self.GRID_SIZE), 1)
    for col in range(5):
      pygame.draw.line(self.screen, self.WHITE, ((self.GRID_WIDTH + 1 + col) * self.GRID_SIZE, 2 * self.GRID_SIZE), ((self.GRID_WIDTH + 1 + col) * self.GRID_SIZE, 6 * self.GRID_SIZE), 1)

    self.draw_shape(self.next_shape, self.GRID_WIDTH + 1, 2)

    for row in range(self.GRID_HEIGHT):
      for col in range(self.GRID_WIDTH):
        if self.grid[row][col] != 0:
          pygame.draw.rect(self.screen, self.COLOURS[self.grid[row][col]], (col * self.GRID_SIZE, row * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))
          pygame.draw.rect(self.screen, self.BLACK, (col * self.GRID_SIZE, row * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE), 2)

  def draw_shape(self, shape, x, y):
    for row in range(len(shape[0])):
      for col in range(len(shape[0][row])):
        if shape[0][row][col] != 0:
          pygame.draw.rect(self.screen,  self.COLOURS[shape[1]], ((x + col) * self.GRID_SIZE, (y + row) * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))
          pygame.draw.rect(self.screen, self.BLACK, ((x + col) * self.GRID_SIZE, (y + row) * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE), 2)

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



Tetris().play_tetris()
# Tetris(Smart_AI()).play_tetris()
# Tetris(RandomAI()).play_tetris()

