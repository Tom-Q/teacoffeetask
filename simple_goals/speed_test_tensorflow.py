# A speed test for tensorflow.
import tensorflow as tf
import numpy as np
import timeit

# Compute error + backprop.
x = np.asarray([[[0, 1] * 10], [[1, 0] * 10]], dtype=np.float32)
size_input = x[0].size
y = np.asarray([[[1]], [[0]]], dtype=np.float32)
size_output = y[0].size # =1
size_hidden = 1000

hidden_layer = tf.Variable(np.random.normal(0., .1, size=[size_input, size_hidden]), dtype=tf.float32)
output_layer = tf.Variable(np.random.normal(0., .1, size=[size_hidden, size_output]), dtype=tf.float32)
recurrent_layer = tf.Variable(np.random.normal(0., .1, size=[size_hidden, size_hidden]), dtype=tf.float32)
all_weights_normal = [hidden_layer, output_layer]
all_weights_recurrent = [hidden_layer, output_layer, recurrent_layer]
learning_rate = 0.1

def multi_speed_test():
    with tf.device('/gpu:0'):
        print("warm-up:", speed_test(100)/100) # run a first time just to warm tensorflow up
        test_100 = speed_test(100) # a second time to get that initialization overhead
        print("post warm-up:", test_100/100)
        print("1000 iter:", (speed_test(1100)-test_100)/1000)
        #print("10000 iter:", (speed_test(10100)-test_100)/10000)

        print("recurrent warm-up:", speed_test_recurrent(100)/100) # run a first time just to warm tensorflow up
        test_100 = speed_test_recurrent(100) # a second time to get that initialization overhead
        print("post warm-up recurrent:", test_100/100)
        print("1000 iter recurrent:", (speed_test_recurrent(1100)-test_100)/1000)
        #print("10000 iter:", (speed_test_recurrent(10100)-test_100)/10000)

        print("warm-up recurrent with tf:", speed_test_recurrent_tf(100)/100) # run a first time just to warm tensorflow up
        test_100 = speed_test_recurrent_tf(100) # a second time to get that initialization overhead
        print("post warm-up recurrent with tf:", test_100/100)
        print("1000 iter  recurrent with tf:", (speed_test_recurrent_tf(1100)-test_100)/1000)
        #print("10000 iter:", (speed_test_recurrent_tf(10100)-test_100)/10000)

    with tf.device('/cpu:0'):
        print("cpu warm-up:", speed_test(100) / 100)  # run a first time just to warm tensorflow up
        test_100 = speed_test(100)  # a second time to get that initialization overhead
        print("cpu post warm-up:", test_100 / 100)
        print("cpu 1000 iter:", (speed_test(1100) - test_100) / 1000)
        # print("10000 iter:", (speed_test(10100)-test_100)/10000)

        print("cpu recurrent warm-up:", speed_test_recurrent(100) / 100)  # run a first time just to warm tensorflow up
        test_100 = speed_test_recurrent(100)  # a second time to get that initialization overhead
        print("cpu post warm-up recurrent:", test_100 / 100)
        print("cpu 1000 iter recurrent:", (speed_test_recurrent(1100) - test_100) / 1000)
        # print("10000 iter:", (speed_test_recurrent(10100)-test_100)/10000)

        print("cpu warm-up recurrent with tf:",
              speed_test_recurrent_tf(100) / 100)  # run a first time just to warm tensorflow up
        test_100 = speed_test_recurrent_tf(100)  # a second time to get that initialization overhead
        print("cpu post warm-up recurrent with tf:", test_100 / 100)
        print("cpu 1000 iter  recurrent with tf:", (speed_test_recurrent_tf(1100) - test_100) / 1000)
        # print("10000 iter:", (speed_test_recurrent_tf(10100)-test_100)/10000)

def speed_test(num_iterations):
    start = timeit.default_timer()
    for i in range(num_iterations):
        exemplar = np.random.randint(0, 2)
        with tf.GradientTape() as tape:
            hl_sums = tf.matmul(x[exemplar], hidden_layer)
            hl_activation = tf.sigmoid(hl_sums)
            ol_activation = tf.matmul(hl_activation, output_layer)
            loss = tf.reduce_sum(tf.square((y[exemplar] - ol_activation)))
            gradients = tape.gradient(loss, all_weights_normal)
        for j in range(len(all_weights_normal)):
            all_weights_normal[j].assign_sub(gradients[j] * learning_rate)
    end = timeit.default_timer()
    return end-start

def speed_test_recurrent(num_iterations):
    start = timeit.default_timer()
    for i in range(num_iterations):
        exemplar = np.random.randint(0, 2)
        prev_hidden_layer = np.zeros((1, size_hidden), dtype=np.float32)
        with tf.GradientTape() as tape:
            for i in range(5):
                hl_sums = tf.matmul(x[exemplar], hidden_layer) + tf.matmul(prev_hidden_layer, recurrent_layer)
                hl_activation = tf.sigmoid(hl_sums)
                prev_hidden_layer = hl_activation
                ol_activation = tf.matmul(hl_activation, output_layer)
            loss = tf.reduce_sum(tf.square((y[exemplar] - ol_activation)))
            gradients = tape.gradient(loss, all_weights_recurrent)
        for j in range(len(all_weights_recurrent)):
            all_weights_recurrent[j].assign_sub(gradients[j] * learning_rate)
    end = timeit.default_timer()
    return end-start


def speed_test_recurrent_tf(num_iterations):
    start = timeit.default_timer()
    for i in range(num_iterations):
        exemplar = np.random.randint(0, 2)
        prev_hidden_layer = np.zeros((1, size_hidden), dtype=np.float32)
        with tf.GradientTape() as tape:
            loss = process(x[exemplar], y[exemplar], tf.convert_to_tensor(prev_hidden_layer))
            gradients = tape.gradient(loss, all_weights_recurrent)
        for j in range(len(all_weights_recurrent)):
            all_weights_recurrent[j].assign_sub(gradients[j] * learning_rate)
    end = timeit.default_timer()
    return end-start


@tf.function
def process(x, y, prev_hidden_layer):
    for i in range(5):
        hl_sums = tf.matmul(x, hidden_layer) + tf.matmul(prev_hidden_layer, recurrent_layer)
        hl_activation = tf.sigmoid(hl_sums)
        prev_hidden_layer = hl_activation
        ol_activation = tf.matmul(hl_activation, output_layer)
    loss = tf.reduce_sum(tf.square((y - ol_activation)))
    return loss