import abc

class BaseManager(abc.ABC):

    def __init__(self, seed, device, **kwargs):
        self.seed = seed
        self.device = device

    @abc.abstractmethod
    def env_rollout(self, agent, env, n_episodes: int, **kwargs):
        pass

    @abc.abstractmethod
    def get_train_and_test_datasets(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_scaler(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_score_function(self, **kwargs):
        pass

    def preprocess_data(self, batch_data):
        return batch_data
