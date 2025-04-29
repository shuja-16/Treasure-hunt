# import pygame
# import sys
# import random

# # === Constants ===
# GRID_SIZE = 10
# TILE_SIZE = 64
# WIDTH = HEIGHT = GRID_SIZE * TILE_SIZE
# HUD_HEIGHT = 60
# FPS = 60

# # === Colors ===
# WHITE = (245, 245, 245)
# BLACK = (0, 0, 0)
# GRAY = (200, 200, 200)
# GREEN = (34, 177, 76)
# RED = (200, 50, 50)
# BLUE = (30, 144, 255)

# # === Assets ===
# pygame.init()
# screen = pygame.display.set_mode((WIDTH, HEIGHT + HUD_HEIGHT))
# pygame.display.set_caption("Treasure Hunt")
# clock = pygame.time.Clock()
# font = pygame.font.SysFont("arial", 24)

# def blit_centered(img, x, y):
#     img_rect = img.get_rect()
#     img_x = x * TILE_SIZE + (TILE_SIZE - img_rect.width) // 2
#     img_y = y * TILE_SIZE + (TILE_SIZE - img_rect.height) // 2
#     screen.blit(img, (img_x, img_y))


# def load_and_scale_image(path):
#     img = pygame.image.load(path).convert_alpha()
#     img = pygame.transform.smoothscale(img, (TILE_SIZE - 16, TILE_SIZE - 16))  # shrink slightly for padding
#     return img

# player_img = load_and_scale_image("assets/player.png")
# treasure_img = load_and_scale_image("assets/treasure.png")
# trap_img = load_and_scale_image("assets/trap.png")

# click_sound = pygame.mixer.Sound("assets/click.wav")
# found_sound = pygame.mixer.Sound("assets/found.wav")
# trap_sound = pygame.mixer.Sound("assets/trap.wav")

# # === Game States ===
# MENU, PLAYING, GAME_OVER = "MENU", "PLAYING", "GAME_OVER"

# # === Game Classes ===
# class Player:
#     def __init__(self):
#         self.x, self.y = 0, 0
#         self.score = 0
#         self.hints = 3

#     def move(self, dx, dy):
#         nx, ny = self.x + dx, self.y + dy
#         if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
#             self.x, self.y = nx, ny

# class Game:
#     def __init__(self):
#         self.player = Player()
#         self.treasures = self.generate_items(5)
#         self.traps = self.generate_items(7)
#         self.revealed_treasures = set()
#         self.revealed_traps = set()
#         self.turns = 30
#         self.hint_msg = ""

#     def generate_items(self, count):
#         items = set()
#         while len(items) < count:
#             x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
#             if (x, y) != (0, 0):
#                 items.add((x, y))
#         return list(items)

#     def update(self):
#         pos = (self.player.x, self.player.y)
#         if pos in self.treasures and pos not in self.revealed_treasures:
#             self.revealed_treasures.add(pos)
#             self.player.score += 1
#             found_sound.play()
#         if pos in self.traps and pos not in self.revealed_traps:
#             self.revealed_traps.add(pos)
#             self.player.x, self.player.y = 0, 0
#             trap_sound.play()
#         self.turns -= 1


#     def use_hint(self):
#         if self.player.hints > 0:
#             self.player.hints -= 1
#             min_dist = float('inf')
#             nearest = None
#             for tx, ty in self.treasures:
#                 if (tx, ty) not in self.revealed_treasures:
#                     dist = abs(tx - self.player.x) + abs(ty - self.player.y)
#                     if dist < min_dist:
#                         min_dist = dist
#                         nearest = (tx, ty)
#             if nearest:
#                 dx = nearest[0] - self.player.x
#                 dy = nearest[1] - self.player.y
#                 parts = []
#                 if dx:
#                     parts.append(f"{abs(dx)} row(s) {'down' if dx > 0 else 'up'}")
#                 if dy:
#                     parts.append(f"{abs(dy)} column(s) {'right' if dy > 0 else 'left'}")
#                 self.hint_msg = "Treasure is " + " and ".join(parts)
#             else:
#                 self.hint_msg = "No treasure nearby."

#     def draw(self):
#         screen.fill(WHITE)

#         # Draw grid
#         for x in range(GRID_SIZE):
#             for y in range(GRID_SIZE):
#                 rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
#                 pygame.draw.rect(screen, GRAY, rect, 1)

#         # Treasures
#         for tx, ty in self.revealed_treasures:
#             blit_centered(treasure_img, tx, ty)

#         # Traps
#         for tx, ty in self.revealed_traps:
#             blit_centered(trap_img, tx, ty)

#         # Player
#         blit_centered(player_img, self.player.x, self.player.y)

#         # HUD
#         pygame.draw.rect(screen, BLACK, (0, HEIGHT, WIDTH, HUD_HEIGHT))
#         draw_text(f"Score: {self.player.score}  Turns: {self.turns}  Hints: {self.player.hints}", 10, HEIGHT+10)
#         draw_text(self.hint_msg, 10, HEIGHT+35)

# # === Helper Functions ===
# def draw_text(text, x, y, color=WHITE):
#     txt = font.render(text, True, color)
#     screen.blit(txt, (x, y))

# def button(text, x, y, w, h):
#     mouse = pygame.mouse.get_pos()
#     click = pygame.mouse.get_pressed()
#     btn_rect = pygame.Rect(x, y, w, h)
#     hover = btn_rect.collidepoint(mouse)
#     pygame.draw.rect(screen, BLUE if hover else GRAY, btn_rect)
#     draw_text(text, x+10, y+10, BLACK)
#     if hover and click[0]:
#         click_sound.play()
#         pygame.time.wait(150)
#         return True
#     return False

# def toggle_fullscreen():
#     pygame.display.toggle_fullscreen()

# # === Main ===
# def main():
#     game_state = MENU
#     game = Game()

#     while True:
#         screen.fill(WHITE)
#         clock.tick(FPS)

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 sys.exit()
#             elif event.type == pygame.KEYDOWN and game_state == PLAYING:
#                 if event.key == pygame.K_UP:
#                     game.player.move(0, -1)
#                 elif event.key == pygame.K_DOWN:
#                     game.player.move(0, 1)
#                 elif event.key == pygame.K_LEFT:
#                     game.player.move(-1, 0)
#                 elif event.key == pygame.K_RIGHT:
#                     game.player.move(1, 0)
#                 elif event.key == pygame.K_h:
#                     click_sound.play()
#                     game.use_hint()
#                 elif event.key == pygame.K_f:
#                     toggle_fullscreen()
#                 game.update()

#         if game_state == MENU:
#             draw_text("Treasure Hunt", WIDTH//2 - 100, HEIGHT//2 - 100, BLACK)
#             if button("Play", WIDTH//2 - 60, HEIGHT//2 - 30, 120, 40):
#                 game = Game()
#                 game_state = PLAYING
#             if button("Quit", WIDTH//2 - 60, HEIGHT//2 + 30, 120, 40):
#                 pygame.quit()
#                 sys.exit()

#         elif game_state == PLAYING:
#             game.draw()
#             if game.turns <= 0 or game.player.score == len(game.treasures):
#                 game_state = GAME_OVER

#         elif game_state == GAME_OVER:
#             draw_text("Game Over!", WIDTH//2 - 70, HEIGHT//2 - 80, RED)
#             draw_text(f"Final Score: {game.player.score}", WIDTH//2 - 90, HEIGHT//2 - 40, BLACK)
#             if button("Play Again", WIDTH//2 - 80, HEIGHT//2, 160, 40):
#                 game = Game()
#                 game_state = PLAYING
#             if button("Exit", WIDTH//2 - 80, HEIGHT//2 + 60, 160, 40):
#                 pygame.quit()
#                 sys.exit()

#         pygame.display.flip()

# if __name__ == "__main__":
#     main()