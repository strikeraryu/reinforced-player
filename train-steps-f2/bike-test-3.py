import pygame as pg
import random
import random
import math
import neat
import pickle
pg.init()

run = True
cell_size = 16
scl = 40
win_size = (scl*cell_size, scl*cell_size)
winner = pickle.load(open("step2.pkl", 'rb'))
config_path = 'config-feedforward.txt'
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_path)
model = neat.nn.FeedForwardNetwork.create(winner, config)

win = pg.display.set_mode(win_size)
pg.display.set_caption("SNAKE GAME")
clock = pg.time.Clock()
white_bike = pg.image.load('white_bike.png')
yellow_bike = pg.image.load('yellow_bike.png')

def distance(x, y, x1, y1):
    ret = math.sqrt((x-x1)**2+(y-y1)**2)
    return ret

class Bike(object):
        def __init__(self, x, y, height, width, colour, head):
            self.x = x
            self.y = y
            self.height = height
            self.width = width
            self.vel = cell_size
            self.dir = "stop"
            self.tail_size = 20
            self.tail_x = [self.x]*self.tail_size
            self.tail_y = [self.y]*self.tail_size
            self.colour = colour
            self.head = head
            self.max_loop = 500 
            self.max_till_now = 0
            self.prev_cords = []

# To draw the snake and the fruit
        def move(self):
            self.tail_x.pop(0)
            self.tail_y.pop(0)
            self.tail_x.append(self.x)
            self.tail_y.append(self.y)
            if self.dir == "up":
                self.y -= self.vel
            elif self.dir == "down":
                self.y += self.vel
            elif self.dir == "left":
                self.x -= self.vel
            elif self.dir == "right":
                self.x += self.vel

            if (self.x, self.y) in self.prev_cords:
                self.max_till_now += 1
            else: self.max_till_now = 0

            self.prev_cords.append((self.x, self.y))
            if len(self.prev_cords) > self.max_loop:
                self.prev_cords.pop(0)

        def draw(self):
            for i in range(self.tail_size):
                pg.draw.rect(
                    win, (self.colour[0], self.colour[1], self.colour[2]), (self.tail_x[i], self.tail_y[i], self.width, self.height))

                dir_head = self.head

                if self.dir == "up":
                    dir_head = pg.transform.rotate(self.head, 180)
                elif self.dir == "down":
                    dir_head = pg.transform.rotate(self.head, 0)
                elif self.dir == "left":
                    dir_head = pg.transform.rotate(self.head, -90)
                elif self.dir == "right":
                    dir_head = pg.transform.rotate(self.head, 90)

                win.blit(dir_head, (self.x, self.y, self.width, self.height))

        def die(self):
            if self.y < 0 or self.y > scl*cell_size-self.height or self.x < 0 or self.x > scl*cell_size-self.width:
               return True, -200

            for i in range(self.tail_size):
                if distance(self.x, self.y, self.tail_x[i], self.tail_y[i]) < cell_size and self.dir!="stop":
                    return True, -350

            if self.max_till_now > self.max_loop:
                return True, -500

            return False, 0

def redrawgamewindow(bikes, bikes_comp):
    win.fill((0, 0, 0))
    for i, bike in enumerate(bikes):
        bike_comp = bikes_comp[i]
        if i < 5:
            bike.draw()
            bike_comp.draw()

    pg.display.update()

def sign(x):
    if x == 0: return 0
    return x/abs(x)

def is_tch(crd, tail, dr):
    x, y = crd
    tx, ty = tail
    dx = tx - x
    dy = ty - y

    if sign(dx) != dr[0] or sign(dy) != dr[1]: return False
    
    if dr[1]==0: a, b, c = 0, 1, -y
    elif dr[0] == 0: a, b, c = 1, 0, -x
    else: a, b, c = -dr[1], dr[0], -dr[0]*y + dr[1]*x

    dst = abs(a*tx + b*ty + c)/(a**2 + b**2)**0.5
    if dst <= 1: return True



def start_test(config_path):        
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)


    winner = p.run(game_loop, 400)

    print('\nBest genome:\n{!s}'.format(winner))
    with open('step3.pkl', 'wb') as output:
        pickle.dump(winner, output)

    pg.quit()
    quit()

def game_loop(genomes, config):
    global run
    bikes = []
    bikes_comp = []
    nets = []
    ge = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        bikes.append(Bike(cell_size * 30, cell_size * 20, cell_size, cell_size, (255, 255, 255), white_bike))
        bikes_comp.append((Bike(cell_size * 20, cell_size * 20, cell_size, cell_size, (255, 255, 0), yellow_bike)))
        g.fitness = 0
        ge.append(g)


    while run and len(bikes) > 0:
        clock.tick(800)

    # To check the events done in the window
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        for ind in range(len(bikes)):
            bike, bike_comp = bikes[ind], bikes_comp[ind]
            window = {}
            move = (0, 0)
            for i in range(-5, 6, 1):
                for j in range(-5, 6, 1):
                    tmp_x, tmp_y = (bike.x - i*cell_size, bike.y - j*cell_size)
                    if tmp_y < 0 or tmp_y > scl*cell_size-bike.height or tmp_x < 0 or tmp_x > scl*cell_size-bike.width:
                        window[(tmp_x, tmp_y)] = 1
                    else: 
                        window[(tmp_x, tmp_y)] = 0

            window[(bike.x, bike.y)] = 2
            if (bike_comp.x, bike_comp.y) in window:
                window[(bike_comp.x, bike_comp.y)] = 3

            for tx, ty in zip(bike.tail_x, bike.tail_y):
                if (tx, ty) in window:
                    window[(tx, ty)] = 4

            for tx, ty in zip(bike_comp.tail_x, bike_comp.tail_y):
                if (tx, ty) in window:
                    window[(tx, ty)] = 5
            
            inputs = list(window.values())

            output = nets[ind].activate(inputs)

            for i, val in enumerate(output):
                move = max((val, i), move)

            if move[1]==0 and bike.dir != "down":
                bike.dir = "up"
            elif move[1]==1 and bike.dir != "up":
                bike.dir = "down"
            elif move[1]==2 and bike.dir != "right":
                bike.dir = "left"
            elif move[1]==3 and bike.dir != "left":
                bike.dir = "right"

        for ind, bike in enumerate(bikes_comp):
            bike_comp = bikes[ind]
            window = {}
            move = (0, 0)
            for i in range(-5, 6, 1):
                for j in range(-5, 6, 1):
                    tmp_x, tmp_y = (bike.x - i*cell_size, bike.y - j*cell_size)
                    if tmp_y < 0 or tmp_y > scl*cell_size-bike.height or tmp_x < 0 or tmp_x > scl*cell_size-bike.width:
                        window[(tmp_x, tmp_y)] = 1
                    else: 
                        window[(tmp_x, tmp_y)] = 0

            window[(bike.x, bike.y)] = 2
            if (bike_comp.x, bike_comp.y) in window:
                window[(bike_comp.x, bike_comp.y)] = 3

            for tx, ty in zip(bike.tail_x, bike.tail_y):
                if (tx, ty) in window:
                    window[(tx, ty)] = 4

            for tx, ty in zip(bike_comp.tail_x, bike_comp.tail_y):
                if (tx, ty) in window:
                    window[(tx, ty)] = 5
            
            inputs = list(window.values())

            output = model.activate(inputs)

            for i, val in enumerate(output):
                move = max((val, i), move)

            if move[1]==0 and bike.dir != "down":
                bike.dir = "up"
            elif move[1]==1 and bike.dir != "up":
                bike.dir = "down"
            elif move[1]==2 and bike.dir != "right":
                bike.dir = "left"
            elif move[1]==3 and bike.dir != "left":
                bike.dir = "right"

        for bike in bikes:
            bike.move()

        for bike in bikes_comp:
            bike.move()

        if run:
            redrawgamewindow(bikes, bikes_comp)
        
        for ind, bike in enumerate(bikes):
            is_die, pen = False, 0
            bike_comp = bikes_comp[ind]
            is_die, pen = bike.die()
            if not is_die: is_die, pen = bike_comp.die(); pen = 0
            if not is_die:
                for i in range(bike_comp.tail_size):
                    if distance(bike.x, bike.y, bike_comp.tail_x[i], bike_comp.tail_y[i]) < cell_size and run:
                        is_die = True
                        pen = -400

                # check if the white bike crushed with yellow bike's tail
                for i in range(bike.tail_size):
                    if distance(bike_comp.x, bike_comp.y, bike.tail_x[i], bike.tail_y[i]) < cell_size and run:
                        is_die = True
                        pen = 400

            if is_die and run:
                ge[ind].fitness += pen
                bikes.pop(ind)
                bikes_comp.pop(ind)
                nets.pop(ind)
                ge.pop(ind)

        for i, bike in enumerate(bikes):
            ge[i].fitness += 1




if __name__ == "__main__":      
    start_test(config_path)