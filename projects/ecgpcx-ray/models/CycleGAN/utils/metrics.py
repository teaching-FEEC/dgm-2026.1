from collections import defaultdict


class LossTracker:

    def __init__(self):

        self.history = defaultdict(list)

    def update(self, **kwargs):

        for key, value in kwargs.items():

            if hasattr(value, "item"):
                value = value.item()

            self.history[key].append(
                float(value)
            )

    def get_history(self):

        return dict(self.history)