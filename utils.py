import datetime
import random


class Stopwatch:
    def start(self):
        self.start_time = self.last_lap = datetime.now()

    def lap(self):
        t = datetime.now()
        delta = t - self.last_lap
        self.last_lap = t
        return delta

    def stop(self):
        return datetime.now() - self.start_time

    def format(self, dt):
        return '{}h {}m {}s'.format(dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)


def roulette_wheel_select(self, array, weights, samples=1):
    suma = sum(weights)
    smp = random.uniform(0, suma, samples)
    results = []
    for x in range(samples):
        acc = 0
        for i in range(len(weights)):
            acc += weights[i]
            if acc >= smp[x]:
                results.append(array[i])
    return results
