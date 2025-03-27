# from garage.experiment import Snapshotter
# import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session

# snapshotter = Snapshotter()
# with tf.compat.v1.Session(): # optional, only for TensorFlow
#     data = snapshotter.load('/home/hisham246/uwaterloo/MetaRL-Assistive-Robotics/data/local/experiment/pearl_trainer_1')
# policy = data['algo'].policy

# print(policy)

import pickle
filename = 'policy_params.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

print(loaded_model)