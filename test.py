import gym
import neat
import os
import math
import random

import numpy as np
import scipy.stats
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.xmeans import kmeans_plusplus_initializer
from pyclustering.utils import draw_clusters

from examples.xor import visualize

env = gym.make('CartPole-v0')
TIMES = 3000
GENERATIONS = 200


def clustering(genomes):
    keys = set()
    for gid, g in genomes:
        for key in g.info.keys():
            keys.add(key)
    keys = sorted(list(keys))
    keys_to_i = {keys[i]: i for i in range(len(keys))}

    ng = len(genomes)
    na = len(keys)
    props = np.zeros((ng, na))
    for i in range(len(genomes)):
        gid, g = genomes[i]
        for key, value in g.info.items():
            props[i][keys_to_i[key]] = random.random()

    props = scipy.stats.zscore(props)
    init_center = kmeans_plusplus_initializer(props, 2).initialize()
    xm = xmeans(props, init_center, ccore=False)
    xm.process()

    clusters = xm.get_clusters()
    draw_clusters(props, clusters)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        observation = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        reward_sum = 0
        for i in range(TIMES):
            opt = net.activate(observation)[0]
            opt = 1 if opt > 0.5 else 0
            observation, reward, done, info = env.step(opt)
            info['A'] = random.randint(20, 40)
            info['B'] = random.random()
            reward_sum += reward
            if abs(observation[0]) > 1:
                reward_sum -= 1
        genome.fitness = reward_sum
        genome.info = info

    clustering(genomes)


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
