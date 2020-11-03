import sys
sys.path.append("..")
import tensorflow as tf
import net_cu as net
import h5py
import input_test as input_data
import get_details as gd
import data_info as di
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


LEARNING_RATE_INIT = 1e-4
DECAY_RATE = 0.99
DECAY_STEP = 2000

MINI_BATCH_SIZE = 256
NUM_CHANNELS = 1

ITER_TIMES = 500000
ITER_TIMES_PER_EVALUATE = 100   # 50次迭代评估一次准确率
ITER_TIME_PER_COUNT_ACCURACY = 20   # 20*32=640 准确率计算分母为640
ITER_TIMES_PER_SAVE = 1000
ITER_TIMES_PER_SAVE_MODEL = 10000

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

######################     getfile: 64x64       ###########################
# IMAGE_SIZE = [[64,64],[32,32],[16,16],[32,16],[8,8],[32,8],[16,8],[8,4],[32,4],[16,4]]
index = 0
CU_WIDTH, CU_HEIGHT, IMAGES_LENGTH, LABEL_LENGTH = gd.get_sample_details(index)

images_batch_train, label_batch_train, qp_batch_train \
                                    = input_data.get_train_dataset(CU_WIDTH, CU_HEIGHT, LABEL_LENGTH, IMAGES_LENGTH)
images_batch_valid, label_batch_valid, qp_batch_valid \
                                    = input_data.get_valid_dataset(CU_WIDTH, CU_HEIGHT, LABEL_LENGTH)
# images_batch_train, label_batch_train, qp_batch_train, sess, iters_train, images_batch_valid, label_batch_valid, qp_batch_valid, iters_valid \
#                                     = input_data.get_data_set(h5f_train, size_train, size_valid, CU_WIDTH, CU_HEIGHT, IMAGES_LENGTH, LABEL_LENGTH, MINI_BATCH_SIZE)


######################     net: 64x64       ###########################

x = tf.placeholder("float", [None, CU_WIDTH, CU_HEIGHT, NUM_CHANNELS])
y = tf.placeholder("float", [None, LABEL_LENGTH])
qp = tf.placeholder("float", [None, 1])
global_step = tf.placeholder("float")

y_probabilty, y_predict, y_one_hot, total_loss_64x64, accuracy_64x64, learning_rate_current, train_step, opt_vars_all, \
    opt_vars_res1,opt_vars_res2 = net.net_64x64(x, y, qp, global_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP)


######################     feed dict       ###########################

saver = tf.train.Saver(opt_vars_all, write_version=tf.train.SaverDef.V2)
saver_res1 = tf.train.Saver(opt_vars_res1, write_version=tf.train.SaverDef.V2)
saver_res2 = tf.train.Saver(opt_vars_res2, write_version=tf.train.SaverDef.V2)

# 或者直接按固定的比例分配。以下代码会占用所有可使用GPU的40%显存。
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.Session(config=config)
# sess = tf.Session()

sess.run(tf.global_variables_initializer())
# sess.run(iters_train.initializer)   # 初始化迭代器
# sess.run(iters_valid.initializer)

for i in range(ITER_TIMES):

    step = i + 1
    feed_step = step

    time_start_input = time.time()

    with tf.device('/gpu:0'):
        images_input, lable_input, qp_input =  sess.run([images_batch_train, label_batch_train, qp_batch_train])

    time_end_input = time.time()

    with tf.device('/gpu:0'):
        _, learning_rate, loss, accuracy = sess.run([train_step, learning_rate_current, total_loss_64x64, accuracy_64x64],feed_dict={x: images_input, y: lable_input, qp: qp_input,global_step: feed_step})
    # y, learning_rate , loss, accuracy =sess.run([y_probabilty, learning_rate_current, total_loss_64x64, accuracy_64x64], feed_dict={x:images_input, y: lable_input, qp:qp_input, global_step: feed_step})

    time_end_calcu = time.time()

    time1 = time_end_input - time_start_input
    time2 = time_end_calcu - time_end_input

    # print(time_start_input , time_end_input, time_end_calcu)
    # print("input time: %d , cal time: %d"%(time1, time2))
    # print(learning_rate, loss, accuracy)

    if step % ITER_TIMES_PER_EVALUATE == 0:
        accuracy_64x64_list = []
        loss_64x64_list = []
        j = 0
        for j in range(ITER_TIME_PER_COUNT_ACCURACY):

            images_input, lable_input, qp_input = sess.run([images_batch_valid, label_batch_valid, qp_batch_valid])
            # 验证时不更新网络参数
            loss, accuracy = sess.run([total_loss_64x64, accuracy_64x64],feed_dict={x: images_input, y: lable_input, qp: qp_input, global_step: feed_step})
            loss_64x64_list.append(loss)
            accuracy_64x64_list.append(accuracy)

        loss = sum(loss_64x64_list[0:ITER_TIME_PER_COUNT_ACCURACY]) / ITER_TIME_PER_COUNT_ACCURACY
        accuracy = sum(accuracy_64x64_list[0:ITER_TIME_PER_COUNT_ACCURACY]) / ITER_TIME_PER_COUNT_ACCURACY
        print("The " + str(step) + " times : loss is " + str(loss) + ",  accuracy is " + str(accuracy) + ", learning_rate is "+str(learning_rate))

        # weight = sess.graph.get_tensor_by_name('w_subnet_nc_64_1')
        # bia = sess.graph.get_tensor_by_name('b_subnet_nc_64_1')
        # w = sess.run(weight)
        # b = sess.run(bia)
        # print(y)

#     if step % ITER_TIMES_PER_SAVE == 0:
#         saver.save(sess, 'Models/ing/model_64x64_%d.dat'%step)
#         saver_res1.save(sess, 'Models/ing/model_res1_%d.dat' % step)
#         saver_res2.save(sess, 'Models/ing/model_res2_%d.dat' % step)
#
# saver.save(sess, 'Models/final/model_64x64.dat')
# saver_res1.save(sess, 'Models/final/model_res1.dat')
# saver_res2.save(sess, 'Models/final/model_res2.dat')