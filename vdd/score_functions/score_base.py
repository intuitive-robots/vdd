class ScoreFunction:
    def __init__(self, model, **kwargs):
        self.model = model

    def __call__(self, samples, states, **kwargs):
        raise NotImplementedError