import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def load_training_data(path):
    with open(path, 'r', encoding = 'utf8') as file:
        next(file)
        lines = [[float(item) for item in line.split(',')] for line in file]
    return (np.array([line[i] for line in lines])[:, None] for i in [0, 1])

x1 = tf.placeholder(tf.float32, [None, 1])
w1 = tf.Variable(tf.random_normal([1, 10]))
b1 = tf.Variable(tf.ones([1, 10]))

x2 = tf.nn.relu(tf.matmul(x1, w1) + b1)
w2 = tf.Variable(tf.random_normal([10, 1]))
b2 = tf.Variable(tf.ones([1, 1]))




# tensorboard_dir = './tensorboard/'
# if not os.path.exists(tensorboard_dir):
#     os.makedirs(tensorboard_dir)

# writer = tf.summary.FileWriter(tensorboard_dir)
# writer.add_graph(session.graph)
predict_values = tf.matmul(x2, w2) + b2
actual_values = tf.placeholder(tf.float32, [None, 1])
loss = tf.reduce_mean(tf.reduce_sum(tf.square(actual_values - predict_values), reduction_indices = [1]))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    (input_data, output_data) = load_training_data('./training_data.dat')
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    for i in range(1000):
        session.run(train_step, feed_dict = {x1:input_data, actual_values:output_data})
        if i % 50 == 0:
            print(session.run(loss, feed_dict = {x1:input_data, actual_values:output_data}))

    # x = input_data
    # y = session.run(predict_values, feed_dict = {x1:x})
    predicted_output_data = session.run(predict_values, feed_dict = {x1:input_data})
    plt.plot(input_data, predicted_output_data)
    plt.plot(input_data, output_data)
    plt.show()

with open('fitting_line.dat', 'w') as file:
    file.write('x, y\n')
    for i in range(input_data.shape[0]):
        file.write('{x}, {y}\n'.format(x = input_data[i, 0], y = predicted_output_data[i, 0]))
