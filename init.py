import tushare as ts
import tensorflow as tf

print(ts.__version__)

print(ts.get_hist_data('600848',start='2019-03-15',end='2019-03-17'))

def build_model(x, keep_prob, y_, output_node_name):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # 28*28*1
    #
    # conv1 = tf.layers.conv2d(x_image, 64, 3, 1, 'same', activation=tf.nn.relu)
    # # 28*28*64
    # pool1 = tf.layers.max_pooling2d(conv1, 2, 2, 'same')
    # # 14*14*64
    #
    # conv2 = tf.layers.conv2d(pool1, 128, 3, 1, 'same', activation=tf.nn.relu)
    # # 14*14*128
    # pool2 = tf.layers.max_pooling2d(conv2, 2, 2, 'same')
    # # 7*7*128
    #
    # conv3 = tf.layers.conv2d(pool2, 256, 3, 1, 'same', activation=tf.nn.relu)
    # # 7*7*256
    # pool3 = tf.layers.max_pooling2d(conv3, 2, 2, 'same')
    # # 4*4*256
    # flatten = tf.reshape(pool3, [-1, 4*4*256])
    # fc = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
    # Convolutional Layer #1
    # tuturial net start
    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # tuturial net end
    # continue
    dropout = tf.nn.dropout(dense, keep_prob)
    logits = tf.layers.dense(dropout, 10)
    outputs = tf.nn.softmax(logits, name=output_node_name)

    # loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

    # train step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    return train_step, loss, accuracy, merged_summary_op


def train(x, keep_prob, y_, train_step, loss, accuracy,
          merged_summary_op, saver):
    print("training start...")

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        tf.train.write_graph(sess.graph_def, OUTPUT_DIR,
                             MODEL_NAME + '.pbtxt', True)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('logs/',
                                               graph=tf.get_default_graph())

        for step in range(NUM_STEPS):
            batch = mnist.train.next_batch(BATCH_SIZE)
            if step % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %f' % (step, train_accuracy))
            _, summary = sess.run([train_step, merged_summary_op],
                                  feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary_writer.add_summary(summary, step)

        saver.save(sess, OUTPUT_DIR + '/' + MODEL_NAME + '.chkp')

        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images,
                                                 y_: mnist.test.labels,
                                                 keep_prob: 1.0})
        print('test accuracy %g' % test_accuracy)

    print("training finished!")