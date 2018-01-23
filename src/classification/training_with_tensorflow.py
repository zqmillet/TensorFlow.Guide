import tensorflow as tf
import numpy as np

def load_training_data(path):
    with open(path, 'r', encoding = 'utf8') as file:
        next(file)
        lines = [[float(item) for item in line.replace(',', ' ').split()] for line in file]
        xs = np.array([line[:2] for line in lines])
        types = np.array([line[2] for line in lines])[:, None]
    return xs, types

def main():
    x = tf.placeholder(tf.float32, [None, 2])
    w = tf.Variable(tf.zeros([2, 1]))
    b = tf.constant([1.])

    predict_type = tf.sigmoid((tf.matmul(x, w) + b) * 10)
    actual_type = tf.placeholder(tf.float32, [None, 1])


    # loss = -tf.reduce_sum(actual_type * tf.log(predict_type) + (1 - actual_type) * tf.log(tf.clip_by_value(1 - predict_type, 1e-9, 1.0)))
    loss = tf.reduce_sum(tf.square(actual_type - predict_type))
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = predict_type, labels = actual_type)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        train_step = tf.train.GradientDescentOptimizer(0.003).minimize(loss)
        xs, types = load_training_data('./training.dat')
        feed_dict = {x:xs, actual_type:types}
        for i in range(1000):
            session.run(train_step, feed_dict = feed_dict)
            if i % 10 == 0: print(session.run(loss, feed_dict = feed_dict))


        print(session.run(w, feed_dict = feed_dict))

if __name__ == '__main__':
    main()