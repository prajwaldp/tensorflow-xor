'''
A simple neural network to solve 2 input XOR
'''

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], "float32")

y = np.array([
    [0],
    [1],
    [1],
    [0]
], "int32")

ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1, random_state=42)


inp = tf.placeholder(tf.float32, shape=[None, 2])
expected = tf.placeholder(tf.float32, shape=[None, 2])


# layer 1 parameters
weights1 = tf.Variable(tf.truncated_normal([2, 3], stddev=0.01))
biases1 = tf.Variable(tf.zeros([3]))
hidden1 = tf.nn.sigmoid(tf.matmul(inp, weights1) + biases1)


# layer 2 (ouput layer) parameters
weights2 = tf.Variable(tf.truncated_normal([3, 2], stddev=0.01))
biases2 = tf.Variable(tf.zeros([2]))
logits = tf.matmul(hidden1, weights2) + biases2


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, expected)
loss = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(expected, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('Cross Entropy Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/2-layer")


init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    print("Training")

    for i in range(5000):
        sess.run(train_step, feed_dict={inp: X_train,
                                        expected: y_train})

        summary, acc, err = sess.run([merged, accuracy, loss], feed_dict={inp: X_train,
                                                                          expected: y_train})

        writer.add_summary(summary, i + 1)

        if (i + 1) % 1000 == 0:
            print("Epoch: {:5d}\tAcc: {:6.2f}%\tErr: {:6.2f}".format(i + 1, acc * 100, err))


    print("\nValidation")
    acc_test = sess.run(accuracy, feed_dict={inp: X_test,
                                             expected: y_test})

    acc_train = sess.run(accuracy, feed_dict={inp: X_train,
                                              expected: y_train})

    print("Accuracy on validation data = {:.2f}%".format(acc_test * 100))
    print("Accuracy on training data = {:.2f}%".format(acc_train * 100))
