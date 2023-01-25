import numpy as np

def random_walk(start, num_steps):
    steps = np.random.normal(loc=0, scale=1, size=num_steps)
    return np.cumsum(np.insert(steps, 0, start))

num_steps = 100
start = 0
time_series = random_walk(start, num_steps)