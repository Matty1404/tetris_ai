import pygame
import random



# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 300, 600
GRID_SIZE = 30
GRID_WIDTH, GRID_HEIGHT = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE   # 10,20
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLOURS = {
            1: pygame.Color(255, 0, 0, a=255), 
            2: pygame.Color(0, 255, 0, a=255), 
            3: pygame.Color(0, 0, 255, a=255),
            4: pygame.Color(255, 255, 0, a=255), 
            5: pygame.Color(255, 0, 255, a=255), 
            6: pygame.Color(0, 255, 255, a=255),
            7: pygame.Color(160, 32, 240, a=255),
            8: pygame.Color(255, 165, 0, a=255)
          }

SHAPES = [
    ([[1, 1, 1, 1]], 1),
    ([[1, 1], [1, 1]], 2),
    ([[1, 1, 1], [0, 1, 0]], 3),
    ([[1, 1, 1], [1, 0, 0]], 4),
    ([[1, 1, 1], [0, 0, 1]], 5),
    ([[1, 1, 1], [1, 1, 1]], 6),
    ([[0, 1, 1], [1, 1, 0]], 7),
    ([[1, 1, 0], [0, 1, 1]], 8),
]

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tetris")

# Initialize clock
clock = pygame.time.Clock()

# Define fonts
font = pygame.font.Font(None, 36)

# Initialize variables
grid = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
current_shape = None
current_x = GRID_WIDTH // 2
current_y = 0
score = 0

# Functions
def draw_grid():
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            if grid[row][col] != 0:
                pygame.draw.rect(screen, COLOURS[grid[row][col]], (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))
                pygame.draw.rect(screen, BLACK, (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE), 2)

def draw_shape(shape, x, y):
    for row in range(len(shape[0])):
        for col in range(len(shape[0][row])):
            if shape[0][row][col] != 0:
                pygame.draw.rect(screen,  COLOURS[shape[1]], ((x + col) * GRID_SIZE, (y + row) * GRID_SIZE, GRID_SIZE, GRID_SIZE))
                pygame.draw.rect(screen, BLACK, ((x + col) * GRID_SIZE, (y + row) * GRID_SIZE, GRID_SIZE, GRID_SIZE), 2)

def check_collision(shape, x, y):
    for row in range(len(shape)):
        for col in range(len(shape[row])):
            if shape[row][col] != 0:
                if x + col < 0 or x + col >= GRID_WIDTH or y + row >= GRID_HEIGHT or grid[y + row][x + col] != 0:
                    return True
    return False

def rotate_shape(shape):
    return [[shape[y][x] for y in range(len(shape))] for x in range(len(shape[0]) - 1, -1, -1)]

def clear_rows():
    global score
    full_rows = []
    for row in range(GRID_HEIGHT):
        if all(grid[row]):
            full_rows.append(row)

    for row in full_rows:
        grid.pop(row)
        grid.insert(0, [0] * GRID_WIDTH)
        score += 100 * len(full_rows)

# Main game loop
game_over = False
current_shape = random.choice(SHAPES)
current_x = GRID_WIDTH // 2
current_y = 0

while not game_over:
  for event in pygame.event.get():
      if event.type == pygame.QUIT:
          game_over = True
      if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_LEFT:
              if not check_collision(current_shape[0], current_x - 1, current_y):
                  current_x -= 1
          elif event.key == pygame.K_RIGHT:
              if not check_collision(current_shape[0], current_x + 1, current_y):
                  current_x += 1
          elif event.key == pygame.K_DOWN:
              if not check_collision(current_shape[0], current_x, current_y + 1):
                  current_y += 1
          elif event.key == pygame.K_UP:
              rotated_shape = rotate_shape(current_shape[0])
              if not check_collision(rotated_shape, current_x, current_y):
                  current_shape = (rotated_shape, current_shape[1])

  # Move the shape down
  if not check_collision(current_shape[0], current_x, current_y + 1):
      current_y += 1
  else:
      # Lock the shape in place
      for row in range(len(current_shape[0])):
          for col in range(len(current_shape[0][row])):
              if current_shape[0][row][col] != 0:
                  grid[current_y + row][current_x + col] = current_shape[1]

      clear_rows()

      # Generate a new random shape
      current_shape = random.choice(SHAPES)
      current_x = GRID_WIDTH // 2
      current_y = 0

      # Check if the game is over
      if check_collision(current_shape[0], current_x, current_y):
          game_over = True

  # Clear the screen
  screen.fill(BLACK)

  # Draw the grid and current shape
  draw_grid()
  draw_shape(current_shape, current_x, current_y)

  # Draw the score
  score_text = font.render(f"Score: {score}", True, WHITE)
  screen.blit(score_text, (10, 10))

  # Update the display
  pygame.display.flip()

  # Limit the frame rate
  clock.tick(5)

# Game over screen
game_over_text = font.render("Game Over", True, WHITE)
screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - game_over_text.get_height() // 2))
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

