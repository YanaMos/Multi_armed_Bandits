import math
import decimal


def ind_max(x):
    m = max(x)
    return x.index(m)


class UCB():
    def __init__(self, counts, values):
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms, counts=0, values=0):
        if counts and values:
            self.counts = counts
            self.values = values

        else:
            self.counts = [0 for col in range(n_arms)]
            self.values = [0.0 for col in range(n_arms)]
        return

    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)

        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = decimal.Decimal(self.values[arm]) + decimal.Decimal(bonus)
        return ind_max(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / decimal.Decimal(n)) * decimal.Decimal(value) + (1 / decimal.Decimal(n)) * decimal.Decimal(reward)
        self.values[chosen_arm] = new_value
        return