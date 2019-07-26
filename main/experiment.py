import argparse
import tensorflow as tf
import os

class Parameters():
    def __init__(self):
        self.dataset_train = {}
        self.dataset_train['input'] = os.path.join("..","datasets", "sparse_train_shuffled.dataset")
        self.dataset_train['label'] = os.path.join("..","datasets", "dense_train_shuffled.dataset")

        self.dataset_val = {}
        self.dataset_val['input'] = os.path.join("..","datasets", "sparse_val.dataset")
        self.dataset_val['label'] = os.path.join("..","datasets", "dense_val.dataset")

        self.dataset_test = {}
        self.dataset_test['input'] = os.path.join("..","datasets", "Htest_velodyne.dataset")
        self.dataset_test['label'] = os.path.join("..","datasets", "Htest_ground.dataset")

        # uncomment for testing on Kitti depth completion test set
        '''
        self.dataset_test['input'] = os.path.join("..","datasets", "submit_test.dataset")
        self.dataset_test['label'] = os.path.join("..","datasets", "submit_test.dataset")
        '''

        self.image_size = (352, 1216)
        self.batch_size = 1

        self.learning_rate = 0.001

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)

        self.invalid_value = 0

        self.max_epochs=16
        self.steps_per_epoch = 85896
        self.steps_per_epoch = self.steps_per_epoch / \
            self.batch_size
        self.num_steps = self.steps_per_epoch * self.max_epochs
        self.log_dir='./../logs'

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

