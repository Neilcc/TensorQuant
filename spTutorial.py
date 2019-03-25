import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tushare as ts
import numpy as np

print(ts.__version__)
ts.get_hist_data('002236', start='2019-02-01',
                 end='2019-03-20').to_csv('dahua.csv')
dahua = ts.get_hist_data('002236', start='2019-02-01', end='2019-03-20')

# filehua = open ('dahua.csv', 'w')
# filehua.write(dahua)
# filehua.close()
f = open('dahua.csv')
df = pd.read_csv(f)
data = np.array(df['high'])
data = data[::-1]
plt.figure()
plt.plot(data)
plt.show()
normalize_data = (data-np.mean(data))/np.std(data)
print(normalize_data)
normalize_data = normalize_data[:, np.newaxis]
print(normalize_data)

time_step = 20
rnn_unit = 10  # hidden layer units
batch_size = 60     # sample size
input_size = 1      # input shape
output_size = 1     # output shape
lr = 0.0006         # learning rate
train_x, train_y = [], []   # set
print train_x
print train_y
for i in range(len(normalize_data)-time_step-1):
    x = normalize_data[i:i+time_step]
    y = normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

# in above we take y [i]as  x[i +1] because we want to predict the price at next day by the price one day before

print train_x
print ' this is y \n'
print train_y

# a node as input second is shape
X = tf.placeholder(tf.float32, [None, time_step, input_size])
# a node as output
Y = tf.placeholder(tf.float32, [None, time_step, output_size])
# random_normal get a tensor that satisfy the given shape in a normal distribution
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}

# the input is the number of datas we work at one time


def lstm(batch):
    # w_in is [inputsize, rnnunit]
    w_in = weights['in']
    # bn is constant node value is 0.1  shape is rnnunit, -1
    b_in = biases['in']
    # reshape X as shap [-1 ,input_size] i.e. [-1, 1]
    input = tf.reshape(X, [-1, input_size])
    # we see a formular a = wx +b
    # here we got  [-1 , inputsize] * [inputsize , rnnunit] = [-1 , runnunit] + bin (bin is rnnunit , -1) then we get what????!!
    input_rnn = tf.matmul(input, w_in)+b_in
    # here we reshape it as -1, timestep rnnunit
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    # create a basic lSTM cell
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # set lstm params  begin state
    init_state = cell.zero_state(batch, dtype=tf.float32)
    # the first ret is outpus  the second is state
    output_rnn, final_states = tf.nn.dynamic_rnn(
        cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    # reshape it to  output  which shape is -1 rnn_unit
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    # here we got  -1, rnn_unit  * rnn unit,1 +===   -1, 1   +   1, -1 again ???
    pred = tf.matmul(output, w_out)+b_out
    return pred, final_states


def train_lstm():
    global batch_size
    pred, _ = lstm(batch_size)
    # reduce
    loss = tf.reduce_mean(
        tf.square(tf.reshape(pred, [-1])-tf.reshape(Y, [-1])))
    # use adam optimizer
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # do it 10k time
        for i in range(10000):
            step = 0
            start = 0
            # every 60 data apart
            end = start+batch_size
            while(end < len(train_x)):
                _, loss_ = sess.run([train_op, loss], feed_dict={
                                    X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end = start+batch_size
                # save data very 10 step
                if step % 10 == 0:
                    print(i, step, loss_)
                    print("save layer:", saver.save(sess, 'stock.model'))
                step += 1
train_lstm()


def prediction():
    pred, _ = lstm(1)  # only[1,time_step,input_size]data inputed
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(base_path+'module2/')
        saver.restore(sess, module_file)
        prev_seq = train_x[-1]
        predict = []
        for i in range(100):
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(
            normalize_data) + len(predict))), predict, color='r')
        plt.show()

prediction()