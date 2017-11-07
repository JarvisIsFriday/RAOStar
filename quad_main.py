#!/usr/bin/env python 

# Visual demo for rao* implementation on quad-with-friend model  
# Yun Chang 2017 
# yunchang@mit.edu

# pygame implementation referenced from mdeyo dstar-lite 

import pygame

# Define some colors
BLACK = (0, 0, 0) 
WHITE = (255, 255, 255)
GREEN = (0, 255, 0) # color of quad 
RED = (255, 0, 0) # color of friend 

colors = {
	0:BLACK,
	1:WHITE
}

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 40
HEIGHT = 40

# This sets the margin between each cell
MARGIN = 5

# Create a 2 dimensional array. A two dimensional
# array is simply a list of lists.
grid = []
for row in range(12):
    # Add an empty array that will hold each cell
    # in this row
    grid.append([])
    for column in range(12):
        grid[row].append(0)  # Append a cell

# Set row 1, cell 5 to one. (Remember rows and
# column numbers start at zero.)
grid[1][5] = 1

# Initialize pygame
pygame.init()

X_DIM = 12
Y_DIM = 12

# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [(WIDTH + MARGIN) * X_DIM + MARGIN,
               (HEIGHT + MARGIN) * Y_DIM + MARGIN]
screen = pygame.display.set_mode(WINDOW_SIZE)

# Set title of screen
pygame.display.set_caption("RAO* Introducing Quad and Friend")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

if __name__ == "__main__":
    basicfont = pygame.font.SysFont('Comic Sans MS', 36)

    # -------- Main Program Loop -----------
    while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # User clicks the mouse. Get the position
                pos = pygame.mouse.get_pos()
                # Change the x/y screen coordinates to grid coordinates
                column = pos[0] // (WIDTH + MARGIN)
                row = pos[1] // (HEIGHT + MARGIN)
                # Set that location to one
                if(grid[row][column] == 0):
                    grid[row][column] = 1

        # Set the screen background
        screen.fill(BLACK)

        # Draw the grid
        for row in range(Y_DIM):
            for column in range(X_DIM):
                color = WHITE

                pygame.draw.rect(screen, colors[grid[row][column]],
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN, WIDTH, HEIGHT])
    

        # # fill in goal cell with GREEN
        # pygame.draw.rect(screen, GREEN, [(MARGIN + WIDTH) * goal_coords[0] + MARGIN,
        #                                  (MARGIN + HEIGHT) * goal_coords[1] + MARGIN, WIDTH, HEIGHT])
        # # print('drawing robot pos_coords: ', pos_coords)
        # # draw moving robot, based on pos_coords
        # robot_center = [int(pos_coords[0] * (WIDTH + MARGIN) + WIDTH / 2) +
        #                 MARGIN, int(pos_coords[1] * (HEIGHT + MARGIN) + HEIGHT / 2) + MARGIN]
        # pygame.draw.circle(screen, RED, robot_center, int(WIDTH / 2) - 2)

        # # draw robot viewing range
        # pygame.draw.rect(
        #     screen, BLUE, [robot_center[0] - VIEWING_RANGE * (WIDTH + MARGIN), robot_center[1] - VIEWING_RANGE * (HEIGHT + MARGIN), 2 * VIEWING_RANGE * (WIDTH + MARGIN), 2 * VIEWING_RANGE * (HEIGHT + MARGIN)], 2)

        # Limit to 60 frames per second
        clock.tick(20)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

    # Be IDLE friendly. If you forget this line, the program will 'hang'
    # on exit.
pygame.quit()