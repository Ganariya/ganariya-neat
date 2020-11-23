import gym
import neat
import os
import math
import random
from examples.xor import visualize

env = gym.make('CartPole-v0')
TIMES = 3000
GENERATIONS = 200


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        observation = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        reward_sum = 0
        for i in range(TIMES):
            opt = net.activate(observation)[0]
            opt = 1 if opt > 0.5 else 0
            observation, reward, done, info = env.step(opt)
            reward_sum += reward
            if abs(observation[0]) > 1:
                reward_sum -= 1
        genome.fitness = reward_sum


def show_result(net):
    env.reset()
    observation = [random.random() for _ in range(4)]
    reward_sum = 0
    for i in range(1000):
        env.render()
        opt = net.activate(observation)[0]
        opt = 1 if opt > 0.5 else 0
        observation, reward, done, info = env.step(opt)
        reward_sum += reward


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, GENERATIONS)
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    show_result(winner_net)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
