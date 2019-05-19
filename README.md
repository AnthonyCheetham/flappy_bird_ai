# Flappy Bird AI

This module is a Reinforcement Learning AI that plays a clone of the *Flappy Bird* app.
The game code was taken from the open source implementation posted [here][codereview], with some small modifications made to accept computer generated inputs and to move some initialization function calls outside of the main loop so it can be run many times in short succession.

The AI itself uses the Q-Learning algorithm to predict the best action (flap or not) at discrete time steps based on 4 variables: X distance to the next pipe, Y distance from the next pipe, the time since the last flap, and an additional variable that indicates whether it is close to the top or bottom of the screen. The Q-table is saved as a numpy save file in qtable.npy. The instance provided with this respository achieves an average score of approximately 130.


## Instructions

To train the AI using 20,000 training games simply run

`python train_ai.py`

To watch a demonstration game, run

`python play_game.py`

While play_game.py is running you can press escape at any time to stop the game.


## Optimizations

Several features of the implementation were tuned to optimize the average score of the AI and minimize the training time.

- The game runs at 30FPS in this instance, but this is still many more opportunities to make decisions than is necessary, and so it
 makes decisions every 4 frames, or every 0.13s.

- The consequences of making a decision affects the game many frames into the future (for example, a pipe gap at the top of the screen may require two or more full flaps to reach, so failing to flap at the first opportunity would put the game in an unwinnable situation that takes many frames to resolve). To speed up the AI's learning, I penalize not only the last decision but the last several decisions whenever it fails. A clear optimum value was found at approximately 1.5 sec (12 states), close to the time taken to reach the highest possible pipe from the bottom of the screen.

- While the game is simulated and displayed on a high resolution grid (568 x 512 pixels), pixel-perfect precision is not needed for the AI, and so a lower-resolution 80 x 160 grid is used. By reducing the number of possible game states, this speeds up the training.

- The 4 variables describing the game state were chosen to give full information to the AI with a minimum number of possible states. The last vector describing proximity to the floor and ceiling was used to force the bird to flap when close to the ground and not flap when close to the ceiling. This allows the consequences of hitting the floor or ceiling to be separate from the consequences of hitting the pipe.

- When training, the AI has a small chance to do the opposite action to the one calculated as the best. This is initialized at 10% and decreased linearly to zero with time. This ensures that the algorithm continues to learn from mistakes and explores state space efficiently.

## Further Improvements

While the current implementation performs well, there are a number of free parameters that can be tuned to achieve better average performance. The learning rate, discount factor, number of frames per decision, resolution in X and Y, and number of penalized states can likely be better tuned. A rough tuning was performed once on each parameter individually, but not globally or following the optimization of other parameters.

[codereview]: http://codereview.stackexchange.com/questions/61477/teaching-a-programming-class-is-my-example-game-well-written

