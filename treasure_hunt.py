# Treasure Hunt Tactics - Final Version (with Minimax + Alpha-Beta Pruning)

import pygame
import random
from collections import defaultdict

# === Game Configuration ===
GRID_SIZE = 10
TILE_SIZE = 50
WIDTH = HEIGHT = GRID_SIZE * TILE_SIZE
FPS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (30, 144, 255)
GREEN = (34, 177, 76)
RED = (237, 28, 36)
GRAY = (200, 200, 200)

# Pygame Initialization
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + 40))
pygame.display.set_caption("Treasure Hunt Tactics")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# === Game Classes ===
class Player:
    def __init__(self):
        self.x, self.y = 0, 0
        self.score = 0
        self.hints = 3
        self.moves = []

    def move(self, dx, dy):
        new_x, new_y = self.x + dx, self.y + dy
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            self.x, self.y = new_x, new_y
            self.moves.append((self.x, self.y))

class Game:
    def __init__(self):
        self.player = Player()
        self.treasures = self.generate_items(5)
        self.traps = self.generate_items(7)
        self.revealed_treasures = set()
        self.revealed_traps = set()
        self.turns_left = 30
        self.moved = False
        self.hint_message = ""

    def generate_items(self, count):
        items = set()
        while len(items) < count:
            x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if (x, y) != (0, 0):
                items.add((x, y))
        return list(items)

    def update(self):
        pos = (self.player.x, self.player.y)
        if pos in self.treasures and pos not in self.revealed_treasures:
            self.revealed_treasures.add(pos)
            self.player.score += 1
        if pos in self.traps and pos not in self.revealed_traps:
            self.revealed_traps.add(pos)
            self.player.x, self.player.y = 0, 0
        self.turns_left -= 1
        self.moved = False

    def use_hint(self):
        if self.player.hints > 0 and self.treasures:
            self.player.hints -= 1
            min_dist = float('inf')
            nearest = None
            for tx, ty in self.treasures:
                if (tx, ty) not in self.revealed_treasures:
                    dist = abs(tx - self.player.x) + abs(ty - self.player.y)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (tx, ty)
            if nearest:
                dx = nearest[0] - self.player.x
                dy = nearest[1] - self.player.y
                msg = []
                if dx != 0:
                    msg.append(f"{abs(dx)} row(s) {'down' if dx > 0 else 'up'}")
                if dy != 0:
                    msg.append(f"{abs(dy)} column(s) {'right' if dy > 0 else 'left'}")
                self.hint_message = "Treasure is " + " and ".join(msg) + "."
            else:
                self.hint_message = "No hints available."

    def draw(self):
        screen.fill(WHITE)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, GRAY, rect, 1)

        for tx, ty in self.revealed_treasures:
            pygame.draw.rect(screen, GREEN, (tx * TILE_SIZE + 5, ty * TILE_SIZE + 5, 40, 40))

        for tx, ty in self.revealed_traps:
            pygame.draw.rect(screen, RED, (tx * TILE_SIZE + 15, ty * TILE_SIZE + 15, 20, 20))

        pygame.draw.circle(screen, BLUE, (self.player.x * TILE_SIZE + 25, self.player.y * TILE_SIZE + 25), 15)

        info_text = font.render(f"Score: {self.player.score}  Turns Left: {self.turns_left}  Hints: {self.player.hints}", True, BLACK)
        screen.blit(info_text, (10, HEIGHT + 5))
        hint_text = font.render(self.hint_message, True, BLACK)
        screen.blit(hint_text, (10, HEIGHT + 20))
        pygame.display.flip()

    def evaluate_state(self, x, y):
        if (x, y) in self.traps:
            return -100
        score = 0
        if (x, y) in self.treasures and (x, y) not in self.revealed_treasures:
            score += 10
        return score

    def minimax(self, x, y, depth, alpha, beta, maximizing):
        if depth == 0:
            return self.evaluate_state(x, y)

        moves = [(0,1),(0,-1),(1,0),(-1,0)]
        if maximizing:
            max_eval = float('-inf')
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    eval = self.minimax(nx, ny, depth-1, alpha, beta, False)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    eval = self.minimax(nx, ny, depth-1, alpha, beta, True)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval

    def get_best_move(self):
        best_score = float('-inf')
        best_move = (0, 0)
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = self.player.x + dx, self.player.y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                score = self.minimax(nx, ny, 2, float('-inf'), float('inf'), False)
                if score > best_score:
                    best_score = score
                    best_move = (dx, dy)
        return best_move

# === Game Loop ===
game = Game()
running = True
while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if not game.moved:
                if event.key == pygame.K_UP:
                    game.player.move(0, -1)
                elif event.key == pygame.K_DOWN:
                    game.player.move(0, 1)
                elif event.key == pygame.K_LEFT:
                    game.player.move(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    game.player.move(1, 0)
                elif event.key == pygame.K_h:
                    game.use_hint()
                game.update()
                game.moved = True

    game.moved = False
    game.draw()

    if game.turns_left <= 0 or game.player.score == len(game.treasures):
        running = False

pygame.quit()
print(f"Game Over! Final Score: {game.player.score}")
