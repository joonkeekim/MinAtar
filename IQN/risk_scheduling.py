import numpy as np


class RiskSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="linear"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        if self.start < self.finish:
            self.delta = (self.finish + self.start) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            if self.start < self.finish:
                return min(self.finish, self.start - self.delta * T)
            else:
                return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))

if __name__=='__main__':
    rs = RiskSchedule(-0.75,0,10)
    risk_schedule = RiskSchedule(1, -0.75, 10000,
                                              decay="linear")

    risk_level = risk_schedule.eval(5000)
    print(risk_level, 1-risk_level)
    # test_lower_risk = test_lower_risk
    # test_upper_risk = test_upper_risk