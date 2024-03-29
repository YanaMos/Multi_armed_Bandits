import UCB
import random


class PredictArm:
    """

    mab:predict_arm
    """

    num_sims = 100
    horizon = 100

    def handle(self):
        data = {111: {'sends': 1000, 'open_rate': 0.11},
                112: {'sends': 1000, 'open_rate': 0.13},
                113: {'sends': 1000, 'open_rate': 0.16}}

        algorithm = UCB.UCB([], [])
        arms = {}
        for subject, subjectData in data.items():
            if subject not in arms:
                arms[subject] = {}

            arms[subject]['value'] = subjectData['open_rate']
            arms[subject]['count'] = subjectData['sends']

        result = self.MabAlgorithm(algorithm, arms)
        print(result)

    def MabAlgorithm(self, algorithm, arms_dict):
        values = list(map(lambda x: x['value'], arms_dict.values()))
        counts = list(map(lambda x: int(x['count'] / 10), arms_dict.values()))
        arms_id = list(arms_dict.keys())
        chosen_arms = [0.0 for i in range(self.num_sims * self.horizon)]
        rewards = [0.0 for i in range(self.num_sims * self.horizon)]
        cumulative_rewards = [0.0 for i in range(self.num_sims * self.horizon)]
        sim_nums = [0.0 for i in range(self.num_sims * self.horizon)]
        times = [0.0 for i in range(self.num_sims * self.horizon)]
        armsResult = {}

        for sim in range(self.num_sims):
            sim = sim + 1
            algorithm.initialize(len(counts), counts=counts, values=values)

            for t in range(self.horizon):
                t = t + 1
                index = (sim - 1) * self.horizon + t - 1
                sim_nums[index] = sim
                times[index] = t

                chosen_arm = algorithm.select_arm()
                chosen_arms[index] = chosen_arm

                reward = self.BernoulliArm(values[chosen_arms[index]])
                rewards[index] = reward

                if arms_id[chosen_arm] not in armsResult:
                    armsResult[arms_id[chosen_arm]] = 0
                else:
                    armsResult[arms_id[chosen_arm]] += 1

                if t == 1:
                    cumulative_rewards[index] = reward
                else:
                    cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

                algorithm.update(chosen_arm, reward)

        return armsResult

    def BernoulliArm(self, probability):
        if random.random() > probability:
            return 0.0
        else:
            return 1.0


