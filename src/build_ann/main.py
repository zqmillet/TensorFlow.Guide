import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

x1 = tf.placeholder(tf.float32, [None, 1])
w1 = tf.Variable(tf.random_normal([1, 10]))
b1 = tf.Variable(tf.ones([1, 10]))

x2 = tf.nn.relu(tf.matmul(x1, w1) + b1)
w2 = tf.Variable(tf.random_normal([10, 1]))
b2 = tf.Variable(tf.ones([1, 1]))

predict_values = tf.matmul(x2, w2) + b2
actual_values = tf.placeholder(tf.float32, [None, 1])
loss = tf.reduce_mean(tf.reduce_sum(tf.square(actual_values - predict_values), reduction_indices = [1]))

session = tf.Session()
session.run(tf.initialize_all_variables())

# tensorboard_dir = './tensorboard/'
# if not os.path.exists(tensorboard_dir):
#     os.makedirs(tensorboard_dir)

# writer = tf.summary.FileWriter(tensorboard_dir)
# writer.add_graph(session.graph)

input_data = np.linspace(-1, 1, 300)[:, None]
output_data = np.square(input_data) + np.random.normal(0, 0.05, input_data.shape)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

for i in range(1000):
    feed_dict = {x1:input_data, actual_values:output_data}
    session.run(train_step, feed_dict = feed_dict)
    if i % 50 == 0:
        print(session.run(loss, feed_dict = feed_dict))

x = input_data
y = session.run(predict_values, feed_dict = {x1:x})

with open('./training_data', 'w') as file:
    for x, y in zip(input_data.tolist(), output_data.tolist()):
        file.write(str(x[0]) + ', ' + str(y[0]) + '\n')

# plt.plot(x, y)
# plt.plot(x, output_data)
# plt.show()
