from game import *
import ai
import pygame


if __name__ == '__main__':
    # If this module had been imported, __name__ would be 'flappybird'.
    # It was executed (e.g. by double-clicking the file), so call main.

    # Do these now to save time    
    display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    images = load_images()

    flappy_bird_AI = ai.AI(silent=True)

    score = main_loop(flappy_bird_AI,silent=True,display=True,display_surface=display_surface,images=images)

    pygame.quit()
    print('Score: {0}'.format(score))