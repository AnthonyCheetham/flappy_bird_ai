from game import *
import ai
import pygame


if __name__ == '__main__':
    # If this module had been imported, __name__ would be 'flappybird'.
    # It was executed (e.g. by double-clicking the file), so call main.
    scores = []
    repeats = 20000

    # Do these now to save time    
    display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    images = load_images()

    flappy_bird_AI = ai.AI(silent=True)
    print('Training AI with {0} iterations'.format(repeats))

    for ix in range(repeats):
        gamble_chance = 0.1*(1-ix/repeats)

        score = main_loop(flappy_bird_AI,silent=True,gamble_chance=gamble_chance,display=False,
            display_surface=display_surface,images=images)
        scores.append(score)
        if (ix % 100) == 99:
            print('  Done {0} of {1}'.format(ix+1,repeats))
            flappy_bird_AI.save_qtable()
            if repeats > 100:
                np.savetxt('last_run.txt',scores)

    # Save the q_table
    flappy_bird_AI.save_qtable()

    pygame.quit()
    print('Best score: {0}'.format(np.max(scores)))
    print('Average score: {0}'.format(np.mean(scores)))
    if repeats > 100:
        np.savetxt('last_run.txt',scores)
