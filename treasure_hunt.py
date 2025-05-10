import pygame
import sys
import random
import math
import numpy as np
from collections import defaultdict
import pickle
from enum import Enum
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import NotFittedError

# === Constants ===
GRID_SIZE = 6
TILE_SIZE = 55
TILE_GAP = 10
GRID_WIDTH = GRID_SIZE * (TILE_SIZE + TILE_GAP) - TILE_GAP
GRID_HEIGHT = GRID_WIDTH
WIDTH = 580
HEIGHT = 580
HUD_HEIGHT = 100
FPS = 10

# Colors
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
BLUE = (30, 144, 255)
GREEN = (50, 200, 50)
DARK_GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)
GRAY = (150, 150, 150)

# Game states
class GameState(Enum):
    MENU = 0
    PLAYING = 1
    GAME_OVER = 2

# === Initialize Pygame ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + HUD_HEIGHT))
pygame.display.set_caption("Treasure Hunt")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 24)

# === Asset Loading ===
def blit_centered(img, x, y):
    img_rect = img.get_rect()
    img_x = (WIDTH - GRID_WIDTH) // 2 + x * (TILE_SIZE + TILE_GAP) + (TILE_SIZE - img_rect.width) // 2
    img_y = (HEIGHT - GRID_HEIGHT) // 2 + y * (TILE_SIZE + TILE_GAP) + (TILE_SIZE - img_rect.height) // 2
    screen.blit(img, (img_x, img_y))

def load_and_scale_image(path):
    try:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, (TILE_SIZE - 16, TILE_SIZE - 16))
    except:
        surface = pygame.Surface((TILE_SIZE-16, TILE_SIZE-16))
        if "player" in path:
            surface.fill(BLUE)
        elif "treasure" in path:
            surface.fill(YELLOW)
        else:
            surface.fill(RED)
        return surface

# Load assets
player_img = load_and_scale_image("assets/player.png")
treasure_img = load_and_scale_image("assets/treasure.png")
trap_img = load_and_scale_image("assets/trap.png")

# Load sounds
try:
    click_sound = pygame.mixer.Sound("assets/click.wav")
    found_sound = pygame.mixer.Sound("assets/found.wav")
    trap_sound = pygame.mixer.Sound("assets/trap.wav")
except:
    # Create dummy sounds if files not found
    click_sound = pygame.mixer.Sound(buffer=bytearray(100))
    found_sound = pygame.mixer.Sound(buffer=bytearray(100))
    trap_sound = pygame.mixer.Sound(buffer=bytearray(100))

class Player:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x, self.y = 0, 0
        self.score = 0
        self.lives = 3
        self.hints = 3
        
    def move(self, dx, dy):
        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            self.x, self.y = nx, ny
            return True
        return False

class TrapRLAgent:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            learning_rate_init=0.01,
            max_iter=1,
            warm_start=True,
            random_state=42
        )
        # Initialize with dummy data
        dummy_X = np.zeros((1, grid_size * grid_size))
        dummy_y = np.zeros((1, grid_size * grid_size))  # Output shape matches input
        self.model.fit(dummy_X, dummy_y)
        
        self.exploration_rate = 0.3
        self.discount_factor = 0.95
        self.last_state = None
        self.last_action = None
        self.model_file = "trap_rl_model.pkl"
        
    def get_state(self, game):
        state = np.zeros(self.grid_size * self.grid_size)
        state[game.player.x * self.grid_size + game.player.y] = 1
        state[game.rival.x * self.grid_size + game.rival.y] = 0.8
        
        for tx, ty in game.treasures:
            idx = tx * self.grid_size + ty
            state[idx] = 0.5 if (tx, ty) in game.revealed_treasures else 0.3
                
        for tx, ty in game.traps:
            idx = tx * self.grid_size + ty
            state[idx] = 0.7
                
        return state
    
    def choose_action(self, game):
        possible_actions = []
        action_indices = []
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pos = (x, y)
                if (x not in [game.player.x, game.rival.x] or 
                    y not in [game.player.y, game.rival.y]) and \
                pos not in game.treasures and pos not in game.traps:
                    possible_actions.append(pos)
                    action_indices.append(x * self.grid_size + y)
        
        if not possible_actions:
            return None
            
        if random.random() < self.exploration_rate:
            return random.choice(possible_actions)
            
        state = self.get_state(game)
        try:
            q_values = self.model.predict(state.reshape(1, -1))
            # Ensure q_values is a 1D array
            q_values = q_values.flatten()
            
            # Get Q-values for possible actions only
            action_q_values = q_values[action_indices]
            best_idx = np.argmax(action_q_values)
            self.last_state = state
            self.last_action = action_indices[best_idx]
            
            return possible_actions[best_idx]
        except NotFittedError:
            return random.choice(possible_actions)
    
    def learn(self, reward):
        if self.last_state is None or self.last_action is None:
            return
            
        try:
            # Get current predictions
            current_q = self.model.predict(self.last_state.reshape(1, -1))
            current_q = current_q.flatten()
            
            # Update only the Q-value for the taken action
            target = current_q.copy()
            target[self.last_action] = reward + self.discount_factor * np.max(current_q)
            
            # Train with the updated target
            self.model.partial_fit(
                self.last_state.reshape(1, -1),
                target.reshape(1, -1)
            )
            
            self.exploration_rate = max(0.05, self.exploration_rate * 0.995)
        except NotFittedError:
            pass
    
    def save_model(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self):
        try:
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
        except:
            pass

class RivalAI:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()
        
    def reset(self):
        self.x = random.randint(0, self.grid_size-1)
        self.y = random.randint(0, self.grid_size-1)
        self.score = 0
        self.search_depth = 3
        
    def make_move(self, game):
        best_score = -math.inf
        best_move = (0, 0)
        alpha = -math.inf
        beta = math.inf
        
        moves = [(0,1), (1,0), (0,-1), (-1,0)]
        random.shuffle(moves)
        
        for dx, dy in moves:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                old_pos = self.x, self.y
                self.x, self.y = nx, ny
                
                score = self.minimax(game, self.search_depth-1, alpha, beta, False)
                
                self.x, self.y = old_pos
                
                if score > best_score:
                    best_score = score
                    best_move = (dx, dy)
                    alpha = max(alpha, best_score)
        
        self.x += best_move[0]
        self.y += best_move[1]
        
        if (self.x, self.y) in game.treasures and (self.x, self.y) not in game.revealed_treasures:
            game.revealed_treasures.add((self.x, self.y))
            self.score += 1
            found_sound.play()
            return True
        return False

    def minimax(self, game, depth, alpha, beta, is_maximizing):
        if depth == 0 or game.is_terminal_state():
            return self.evaluate(game, is_maximizing)
            
        if is_maximizing:
            max_eval = -math.inf
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    old_pos = self.x, self.y
                    self.x, self.y = nx, ny
                    eval = self.minimax(game, depth-1, alpha, beta, False)
                    self.x, self.y = old_pos
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = math.inf
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                nx, ny = game.player.x + dx, game.player.y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    old_pos = game.player.x, game.player.y
                    game.player.x, game.player.y = nx, ny
                    eval = self.minimax(game, depth-1, alpha, beta, True)
                    game.player.x, game.player.y = old_pos
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval

    def evaluate(self, game, is_maximizing):
        score = 0
        
        # Base scores
        score += self.score * 100
        
        # Treasure proximity
        for tx, ty in game.treasures:
            if (tx, ty) not in game.revealed_treasures:
                dist = abs(tx - self.x) + abs(ty - self.y)
                score += (GRID_SIZE - dist) * 10
                
                # Blocking bonus
                player_dist = abs(tx - game.player.x) + abs(ty - game.player.y)
                if dist < player_dist:
                    score += 5
        
        # Threat assessment
        player_dist = abs(game.player.x - self.x) + abs(game.player.y - self.y)
        score += (GRID_SIZE - player_dist) * 3
        
        # Penalize dangerous positions
        if not is_maximizing:
            for tx, ty in game.traps:
                if (tx, ty) not in game.revealed_traps:
                    trap_dist = abs(tx - self.x) + abs(ty - self.y)
                    if trap_dist <= 1:
                        score -= 20 * (2 - trap_dist)
        
        return score

class Game:
    def __init__(self):
        self.player = Player()
        self.rival = RivalAI(GRID_SIZE)
        self.trap_agent = TrapRLAgent(GRID_SIZE)
        self.treasures = []
        self.traps = []
        self.revealed_treasures = set()
        self.revealed_traps = set()
        self.time_left = 60
        self.start_time = 0
        self.game_state = GameState.MENU
        self.message = ""
        self.reset()
        
    def reset(self):
        self.player.reset()
        self.rival.reset()
        self.trap_agent.load_model()
        self.treasures = self._generate_items(5)
        self.traps = self._generate_items(3)
        self.revealed_treasures = set()
        self.revealed_traps = set()
        self.start_time = pygame.time.get_ticks()
        self.time_left = 60
        self.message = ""
    
    def _generate_items(self, count):
        items = set()
        while len(items) < count:
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            if (x, y) != (0, 0):
                items.add((x, y))
        return list(items)
    
    def is_terminal_state(self):
        if len(self.revealed_treasures) == len(self.treasures):
            self.message = "You collected all treasures!" if self.player.score > self.rival.score else "Rival collected more treasures!"
            return True
            
        if self.time_left <= 0:
            self.message = "Time ran out!"
            return True
        if self.player.lives <= 0:
            self.message = "You ran out of lives!"
            return True
            
        return False
    
    def update(self):
        # Rival makes move
        self.rival.make_move(self)
        
        # Check player position
        pos = (self.player.x, self.player.y)
        
        # Treasure collection
        if pos in self.treasures and pos not in self.revealed_treasures:
            self.revealed_treasures.add(pos)
            self.player.score += 1
            found_sound.play()
            self.trap_agent.learn(-1.0)  # Negative reward
            
        # Trap collision
        if pos in self.traps and pos not in self.revealed_traps:
            self.revealed_traps.add(pos)
            self.player.x, self.player.y = 0, 0
            self.player.lives -= 1
            trap_sound.play()
            self.trap_agent.learn(1.0)  # Positive reward
            
        # Add new traps periodically (RL-driven)
        if random.random() < 0.1 and len(self.traps) < 7:
            new_trap = self.trap_agent.choose_action(self)
            if new_trap and new_trap not in self.traps:
                self.traps.append(new_trap)
                self.trap_agent.learn(-0.1)  # Small penalty
                
        # Update time
        self.time_left = max(0, 60 - (pygame.time.get_ticks() - self.start_time) // 1000)
        
        return self.is_terminal_state()
    
    def draw(self):
        # Draw background
        screen.fill(DARK_GRAY)
        
        # Draw grid
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(
                    (WIDTH - GRID_WIDTH) // 2 + x * (TILE_SIZE + TILE_GAP),
                    (HEIGHT - GRID_HEIGHT) // 2 + y * (TILE_SIZE + TILE_GAP),
                    TILE_SIZE, TILE_SIZE
                )
                pygame.draw.rect(screen, (240, 230, 200), rect, border_radius=5)
                
                # Highlight nearby traps
                if (x, y) in self.traps and (x, y) not in self.revealed_traps:
                    if abs(x - self.player.x) <= 1 and abs(y - self.player.y) <= 1:
                        pygame.draw.rect(screen, (200, 100, 100), rect, 2, border_radius=5)
        
        # Draw game elements
        for tx, ty in self.revealed_treasures:
            blit_centered(treasure_img, tx, ty)
        
        for tx, ty in self.revealed_traps:
            blit_centered(trap_img, tx, ty)
        
        # Draw characters
        blit_centered(player_img, self.player.x, self.player.y)
        
        # Draw rival (as red square)
        rival_rect = pygame.Rect(
            (WIDTH - GRID_WIDTH) // 2 + self.rival.x * (TILE_SIZE + TILE_GAP) + TILE_SIZE//4,
            (HEIGHT - GRID_HEIGHT) // 2 + self.rival.y * (TILE_SIZE + TILE_GAP) + TILE_SIZE//4,
            TILE_SIZE//2, TILE_SIZE//2
        )
        pygame.draw.rect(screen, RED, rival_rect, border_radius=3)
        
        # Draw HUD
        pygame.draw.rect(screen, BLACK, (0, HEIGHT, WIDTH, HUD_HEIGHT))
        
        # HUD text
        texts = [
            f"Player: {self.player.score}",
            f"Rival: {self.rival.score}",
            f"Lives: {self.player.lives}",
            f"Time: {self.time_left}s",
            f"Hints: {self.player.hints}"
        ]
        
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, WHITE)
            screen.blit(text_surface, (10 + i * 120, HEIGHT + 20))

def draw_text(text, x, y, color=WHITE, size=24):
    font = pygame.font.SysFont("arial", size)
    txt = font.render(text, True, color)
    screen.blit(txt, (x, y))

def button(text, x, y, w, h):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    btn_rect = pygame.Rect(x, y, w, h)
    hover = btn_rect.collidepoint(mouse)
    pygame.draw.rect(screen, BLUE if hover else GRAY, btn_rect, border_radius=10)
    draw_text(text, x + 10, y + 10, BLACK)
    if hover and click[0]:
        click_sound.play()
        pygame.time.wait(150)
        return True
    return False

def main():
    game = Game()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if game.game_state == GameState.MENU:
                if event.type == pygame.KEYDOWN:
                    game.game_state = GameState.PLAYING
                    click_sound.play()
                    
            elif game.game_state == GameState.PLAYING:
                if event.type == pygame.KEYDOWN:
                    moved = False
                    if event.key == pygame.K_UP:
                        moved = game.player.move(0, -1)
                    elif event.key == pygame.K_DOWN:
                        moved = game.player.move(0, 1)
                    elif event.key == pygame.K_LEFT:
                        moved = game.player.move(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        moved = game.player.move(1, 0)
                    elif event.key == pygame.K_h and game.player.hints > 0:
                        game.player.hints -= 1
                        click_sound.play()
                        
                    if moved and game.update():
                        game.game_state = GameState.GAME_OVER
                        game.trap_agent.save_model()
                        
            elif game.game_state == GameState.GAME_OVER:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        game.reset()
                        game.game_state = GameState.PLAYING
                        click_sound.play()
                    elif event.key == pygame.K_q:
                        running = False
        
        # Drawing
        screen.fill(BLACK)
        
        if game.game_state == GameState.MENU:
            draw_text("TREASURE HUNT", WIDTH//2 - 120, HEIGHT//2 - 60, BLUE, 48)
            draw_text("AI EDITION", WIDTH//2 - 80, HEIGHT//2, WHITE, 32)
            draw_text("Press any key to start", WIDTH//2 - 100, HEIGHT//2 + 60)
            
        elif game.game_state == GameState.PLAYING:
            game.draw()
            
        elif game.game_state == GameState.GAME_OVER:
            screen.fill(DARK_GRAY)
            
            if len(game.revealed_treasures) == len(game.treasures) and game.player.score > game.rival.score:
                draw_text("YOU WON!", WIDTH//2 - 80, HEIGHT//2 - 60, GREEN, 48)
            else:
                draw_text("GAME OVER", WIDTH//2 - 100, HEIGHT//2 - 60, RED, 48)
                
            draw_text(game.message, WIDTH//2 - 120, HEIGHT//2 - 10)
            draw_text(f"Player: {game.player.score} | Rival: {game.rival.score}", WIDTH//2 - 100, HEIGHT//2 + 30)
            draw_text("Press R to restart or Q to quit", WIDTH//2 - 120, HEIGHT//2 + 80)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    game.trap_agent.save_model()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()