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
GRID_OFFSET_X = (WIDTH - GRID_WIDTH) // 2
GRID_OFFSET_Y = (HEIGHT - GRID_HEIGHT) // 2

# Colors
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
BLUE = (30, 144, 255)
GREEN = (50, 200, 50)
DARK_GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)
GOLD = (255, 215, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GRAY = (150, 150, 150)
SKIN_COLOR = (255, 224, 189)
LIGHT_BLUE = (173, 216, 230)

DARK_BG = (30, 30, 40)  # Dark gray-blue background
DARKER_BG = (20, 20, 30)  # Even darker for overlays
ACCENT_COLOR = (36, 174, 93)  # Light blue for accents
SECONDARY_COLOR = (150, 150, 180)  # Light gray for secondary text
BUTTON_COLOR = (60, 60, 80)  # Dark gray for buttons
BUTTON_HOVER = (80, 80, 100)  # Lighter gray for button hover
GREEN = (50, 200, 50)  # Green for positive actions
RED = (200, 50, 50)  # Red for negative actions

# Game states
class GameState(Enum):
    MENU = 0
    PLAYING = 1
    GAME_OVER = 2
    LEVEL_COMPLETE = 3
    CONGRATULATIONS = 4
    LEVEL_SELECT = 5 

# === Initialize Pygame ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + HUD_HEIGHT))
pygame.display.set_caption("Treasure Hunt Adventure")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 24)
title_font = pygame.font.SysFont("arial", 48, bold=True)
subtitle_font = pygame.font.SysFont("arial", 36)

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
player_img = load_and_scale_image("assets/player2.png")
treasure_img = load_and_scale_image("assets/treasure.png")
trap_img = load_and_scale_image("assets/trap.png")
rival_img = load_and_scale_image("assets/rival.png")

try:
    background_img = pygame.image.load("assets/background.png")
    background_img = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
except:
    background_img = pygame.Surface((WIDTH, HEIGHT))
    background_img.fill((34, 139, 34))  # fallback green background

# Load sounds
try:
    click_sound = pygame.mixer.Sound("assets/click.wav")
    found_sound = pygame.mixer.Sound("assets/found.wav")
    trap_sound = pygame.mixer.Sound("assets/trap.wav")
    victory_sound = pygame.mixer.Sound("assets/found.wav")
except:
    click_sound = pygame.mixer.Sound(buffer=bytearray(100))
    found_sound = pygame.mixer.Sound(buffer=bytearray(100))
    trap_sound = pygame.mixer.Sound(buffer=bytearray(100))
    victory_sound = pygame.mixer.Sound(buffer=bytearray(100))

class Player:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x, self.y = 0, 0
        self.score = 0
        self.lives = 3
        
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
        if game.level == 2:
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
                if (x not in [game.player.x] or y not in [game.player.y]) and \
                pos not in game.treasures and pos not in game.traps:
                    if game.level == 1 or (game.level == 2 and (x not in [game.rival.x] or y not in [game.rival.y])):
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
        self.time_left = 40
        self.start_time = 0
        self.game_state = GameState.MENU
        self.message = ""
        self.level = 1
        self.transition_timer = 0
        self.reset()
        
    def reset(self):
        self.player.reset()
        self.rival.reset()
        self.trap_agent.load_model()
        self.treasures = self._generate_items(5)
        self.traps = self._generate_items(3, exclude=self.treasures)
        self.revealed_treasures = set()
        self.revealed_traps = set()
        self.start_time = pygame.time.get_ticks()
        self.time_left = 40 if self.level == 1 else 60
        self.message = ""
        self.transition_timer = 0

    def _generate_items(self, count, exclude=None):
        items = set()
        if exclude is None:
            exclude = []
        while len(items) < count:
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            pos = (x, y)
            if pos != (0, 0) and pos not in exclude:
                items.add(pos)
        return list(items)
    
    def is_terminal_state(self):
        if len(self.revealed_treasures) == len(self.treasures):
            if self.level == 1:
                self.message = "Level 1 Complete! Moving to Level 2"
                self.game_state = GameState.LEVEL_COMPLETE
                return True
            else:
                self.message = "You did it! Victory is yours!" if self.player.score > self.rival.score else "Rival collected more treasures!"
                self.game_state = GameState.GAME_OVER
                return True
                
        if self.time_left <= 0:
            if self.level == 1:
                self.message = "Time ran out! Try again."
            else:
                self.message = "Time ran out!"
            self.game_state = GameState.GAME_OVER
            return True
        if self.player.lives <= 0:
            self.message = "You ran out of lives!"
            self.game_state = GameState.GAME_OVER
            return True
            
        return False
    
    def update(self):
        # In level 2, rival makes move
        if self.level == 2:
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
        if random.random() < 0.1 and len(self.traps) < (5 if self.level == 1 else 7):
            new_trap = self.trap_agent.choose_action(self)
            if new_trap and new_trap not in self.traps:
                self.traps.append(new_trap)
                self.trap_agent.learn(-0.1)  # Small penalty
                
        # Update time
        elapsed = (pygame.time.get_ticks() - self.start_time) // 1000
        self.time_left = max(0, (40 if self.level == 1 else 60) - elapsed)
        
        return self.is_terminal_state()
    
    def draw(self):
        # Draw background
        screen.blit(background_img, (0, 0))
        
        # Draw grid
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect_x = GRID_OFFSET_X + x * (TILE_SIZE + TILE_GAP)
                rect_y = GRID_OFFSET_Y + y * (TILE_SIZE + TILE_GAP)
                tile_rect = pygame.Rect(rect_x, rect_y, TILE_SIZE, TILE_SIZE)
                shadow_rect = tile_rect.move(4, 4)
                pygame.draw.rect(screen, (0, 0, 0, 60), shadow_rect, border_radius=12)
                pygame.draw.rect(screen, (240, 230, 200), tile_rect, border_radius=12)

                if (x, y) in self.traps:
                    neighbors = [(x+dx, y+dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                 if (dx != 0 or dy != 0) and 0 <= x+dx < GRID_SIZE and 0 <= y+dy < GRID_SIZE]
                    if (self.player.x, self.player.y) in neighbors:
                        pygame.draw.rect(screen, RED, tile_rect, 2, border_radius=12)
        
        # Draw game elements
        for tx, ty in self.revealed_treasures:
            blit_centered(treasure_img, tx, ty)
        
        for tx, ty in self.revealed_traps:
            blit_centered(trap_img, tx, ty)
        
        # Draw player
        blit_centered(player_img, self.player.x, self.player.y)
        
        # Draw rival (as red square) in level 2
        if self.level == 2:
            blit_centered(rival_img, self.rival.x, self.rival.y)
        
        # Draw HUD
        pygame.draw.rect(screen, BLACK, (0, HEIGHT, WIDTH, HUD_HEIGHT))
        
        # HUD text
        texts = [
            f"Score: {self.player.score}",
            f"Lives: {self.player.lives}",
            f"Time: {self.time_left}s"
        ]
        
        if self.level == 2:
            texts.insert(1, f"Rival: {self.rival.score}")
        
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, WHITE)
            screen.blit(text_surface, (10 + i * 120, HEIGHT + 20))

def draw_text(text, x, y, color=WHITE, size=24, font_type="regular", center=False):
    if font_type == "title":
        font_obj = title_font
    elif font_type == "subtitle":
        font_obj = subtitle_font
    else:
        font_obj = pygame.font.SysFont("arial", size)
    
    txt = font_obj.render(text, True, color)
    if center:
        txt_rect = txt.get_rect(center=(x, y))
        screen.blit(txt, txt_rect)
    else:
        screen.blit(txt, (x, y))

def draw_button(text, x, y, w, h, base_color, hover_color, text_color):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    btn_rect = pygame.Rect(x, y, w, h)
    hover = btn_rect.collidepoint(mouse)
    
    # Draw button with subtle glow effect
    if hover:
        glow = pygame.Surface((w + 10, h + 10))
        glow.fill((0, 0, 0, 0))
        pygame.draw.rect(glow, (*text_color[:3], 30), glow.get_rect(), border_radius=12)
        screen.blit(glow, (x - 5, y - 5))
    
    pygame.draw.rect(screen, hover_color if hover else base_color, btn_rect, border_radius=10)
    pygame.draw.rect(screen, (100, 100, 120), btn_rect, 2, border_radius=10)  # Border
    
    # Draw text
    draw_text(text, x + w//2, y + h//2 - 2, text_color, 24, "regular", True)
    
    if hover and click[0]:
        click_sound.play()
        pygame.time.wait(150)
        return True
    return False

def draw_animated_background():
    # Dark background with subtle pattern
    screen.fill(DARK_BG)
    for i in range(0, WIDTH, 40):
        for j in range(0, HEIGHT + HUD_HEIGHT, 40):
            if (i + j) % 80 == 0:
                pygame.draw.rect(screen, (40, 40, 50), (i, j, 40, 40), 1)

def main():
    game = Game()
    running = True
    
    while running:
        current_time = pygame.time.get_ticks()
        
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
                        
                    if moved and game.update():
                        if game.game_state == GameState.LEVEL_COMPLETE:
                            game.transition_timer = current_time
                            victory_sound.play()
                            
            elif game.game_state == GameState.CONGRATULATIONS:
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    if current_time - game.transition_timer > 2000:  # 2 second delay
                        game.level = 2
                        game.reset()
                        game.game_state = GameState.PLAYING
                        click_sound.play()
                    
            elif game.game_state == GameState.GAME_OVER:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        game.level = 1
                        game.reset()
                        game.game_state = GameState.PLAYING
                        click_sound.play()
                    elif event.key == pygame.K_q:
                        running = False
        
        # Drawing
        if game.game_state in [GameState.MENU, GameState.GAME_OVER, GameState.CONGRATULATIONS, GameState.LEVEL_COMPLETE]:
            draw_animated_background()
        
        if game.game_state == GameState.MENU:
            # Draw menu with fancy effects
            draw_animated_background()
    
            title_y = HEIGHT//2 - 100
            subtitle_y = HEIGHT//2 - 30
            
            # Animate title position
            title_y += 5 * math.sin(current_time / 300)
            
            draw_text("TREASURE HUNT", WIDTH//2, title_y, ACCENT_COLOR, 45, "title", True)
            draw_text("ADVENTURE", WIDTH//2, title_y + 60, SECONDARY_COLOR, 32, "subtitle", True)
            
            # Draw treasure icon
            treasure_rect = treasure_img.get_rect(center=(WIDTH//2, title_y - 80))
            screen.blit(treasure_img, treasure_rect)
            
            # Draw buttons with dark theme
            btn_y = HEIGHT//2 + 100
            if draw_button("START GAME", WIDTH//2 - 100, btn_y, 200, 45, BUTTON_COLOR, BUTTON_HOVER, ACCENT_COLOR):
                game.game_state = GameState.PLAYING

            if draw_button("LEVEL SELECT", WIDTH//2 - 100, btn_y + 60, 200, 45, BUTTON_COLOR, BUTTON_HOVER, ACCENT_COLOR):
                game.game_state = GameState.LEVEL_SELECT
                
            if draw_button("QUIT", WIDTH//2 - 100, btn_y + 120, 200, 45, BUTTON_COLOR, BUTTON_HOVER, RED):
                running = False
                
            draw_text("Use arrow keys to move", WIDTH//2, HEIGHT + 10, SECONDARY_COLOR, 20, "regular", True)
        elif game.game_state == GameState.PLAYING:
            game.draw()
            draw_text(f"Level {game.level}", WIDTH // 2, GRID_OFFSET_Y - 50, SKIN_COLOR, 30, "regular", True)
            
        elif game.game_state == GameState.LEVEL_COMPLETE:
            # Level complete transition screen
            alpha = min(255, (current_time - game.transition_timer) / 500 * 255)
            overlay = pygame.Surface((WIDTH, HEIGHT + HUD_HEIGHT))
            overlay.fill((0, 0, 0))
            overlay.set_alpha(alpha)
            screen.blit(overlay, (0, 0))
            
            if alpha == 255:
                game.game_state = GameState.CONGRATULATIONS
                game.transition_timer = current_time

        elif game.game_state == GameState.LEVEL_SELECT:
                draw_animated_background()
    
                draw_text("CHOOSE A LEVEL", WIDTH//2, HEIGHT//2 - 100, ACCENT_COLOR, 40, "title", True)

                btn_y = HEIGHT//2 - 20
                if draw_button("LEVEL 1", WIDTH//2 - 110, btn_y, 220, 50, BUTTON_COLOR, BUTTON_HOVER, ACCENT_COLOR):
                    game.level = 1
                    game.reset()
                    game.game_state = GameState.PLAYING

                if draw_button("LEVEL 2", WIDTH//2 - 110, btn_y + 70, 220, 50, BUTTON_COLOR, BUTTON_HOVER, ACCENT_COLOR):
                    game.level = 2
                    game.reset()
                    game.game_state = GameState.PLAYING

                # Back button
                if draw_button("BACK", WIDTH//2 - 110, btn_y + 150, 220, 45, BUTTON_COLOR, BUTTON_HOVER, RED):
                    game.game_state = GameState.MENU
                
        elif game.game_state == GameState.CONGRATULATIONS:
            # Congratulations screen before level 2
            # Dark overlay
            overlay = pygame.Surface((WIDTH, HEIGHT + HUD_HEIGHT))
            overlay.fill(DARKER_BG)
            screen.blit(overlay, (0, 0))
            
            draw_text("CONGRATULATIONS!", WIDTH//2, HEIGHT//2 - 80, ACCENT_COLOR, 45, "title", True)
            draw_text(f"You completed Level {game.level}!", WIDTH//2, HEIGHT//2 - 20, SECONDARY_COLOR, 36, "subtitle", True)
            
            if game.level == 1:
                draw_text("Get ready for Level 2!", WIDTH//2, HEIGHT//2 + 30, SECONDARY_COLOR, 24, "regular", True)
                draw_text("Now you'll face a rival treasure hunter!", WIDTH//2, HEIGHT//2 + 70, SECONDARY_COLOR, 20, "regular", True)
            
            if current_time - game.transition_timer > 2000:
                draw_text("Click to continue...", WIDTH//2, HEIGHT - 50, SECONDARY_COLOR, 20, "regular", True)

                
        elif game.game_state == GameState.GAME_OVER:
            # Dark red overlay
            overlay = pygame.Surface((WIDTH, HEIGHT + HUD_HEIGHT))
            overlay.fill(DARKER_BG)
            screen.blit(overlay, (0, 0))
            
            draw_text("GAME OVER", WIDTH//2, HEIGHT//2 - 100, ACCENT_COLOR, 48, "title", True)
            draw_text(game.message, WIDTH//2, HEIGHT//2 - 40, SECONDARY_COLOR, 24, "regular", True)
            
            if game.level == 2:
                draw_text(f"Your Score: {game.player.score}", WIDTH//2, HEIGHT//2, SECONDARY_COLOR, 24, "regular", True)
                draw_text(f"Rival Score: {game.rival.score}", WIDTH//2, HEIGHT//2 + 30, SECONDARY_COLOR, 24, "regular", True)
            else:
                draw_text(f"Final Score: {game.player.score}", WIDTH//2, HEIGHT//2, SECONDARY_COLOR, 24, "regular", True)
            
            # Draw buttons with dark theme
            btn_y = HEIGHT//2 + 80
            if draw_button("PLAY AGAIN", WIDTH//2 - 220, btn_y, 200, 50, BUTTON_COLOR, BUTTON_HOVER, GREEN):
                game.level = 1
                game.reset()
                game.game_state = GameState.PLAYING
                
            if draw_button("QUIT", WIDTH//2 + 20, btn_y, 200, 50, BUTTON_COLOR, BUTTON_HOVER, RED):
                running = False


        pygame.display.flip()
        clock.tick(FPS)
    
    game.trap_agent.save_model()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()