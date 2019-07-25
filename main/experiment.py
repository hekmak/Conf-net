import argparse

class Parameters():
    def __init__(self):
        self.dataset_train = {}
        self.dataset_train['input'] = None
        self.dataset_train['label'] = None

        self.dataset_val = {}
        self.dataset_val['input'] = None
        self.dataset_val['label'] = None

        self.dataset_test = {}
        self.dataset_test['input'] = None
        self.dataset_test['label'] = None

        self.image_size = None
        self.batch_size = 1

        self.learning_rate = 0.001

        self.optimizer = None

        self.invalid_value = 0

        self.epochs=10
        self.steps_per_epoch=1

        self.log_dir='/logs'

        self.shuffle = True
        self.prefetch_buffer_size = 4

class Experiment(object):

    def __init__(self):
        self.parameters = Parameters()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Completion with Error.')
        subparsers = parser.add_subparsers()

        # parser for training
        train_parser = subparsers.add_parser('train')
        # parser for validation
        val_parser = subparsers.add_parser('val')
        # parser for testing
        test_parser = subparsers.add_parser('test')

        train_parser.set_defaults(func=self.train)
        val_parser.set_defaults(func=self.val)
        test_parser.set_defaults(func=self.test)

        args = parser.parse_args()
        args.func()

        return args

    def run(self):
        self.parse_args()

    def network(self, input, **kwargs):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def val(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

