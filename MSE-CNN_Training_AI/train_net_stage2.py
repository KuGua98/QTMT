import tensorflow as tf
import net_cu as net
import h5py
import input_data_H5 as input_data
import warnings
warnings.filterwarnings("ignore")

# IMAGE_SIZE = [[64,64],[32,32],[16,16],[32,16],[8,8],[32,8],[16,8],[8,4],[32,4],[16,4]]
CU_WIDTH_LIST = [64,  32, 16, 32, 8, 32, 16, 8, 32, 16]
CU_HEIGHT_LIST = [64, 32, 16, 16, 8, 8,  8,  4, 4,   4]

LABEL_LENGTH_LIST = [2, 6, 6, 6, 6, 6, 6, 6, 6, 6]
SAMPLE_LENGTH_LIST = [4105, 1033, 265, 521, 73, 265, 137, 41, 137, 73]
IMAGES_LENGTH_LIST = [4096, 1024, 256, 512, 64, 256, 128, 32, 128, 64]
SAMPLE_AMOUNT_LIST = [600,  0,    0,   0,   0,  0,   0,   0,  0,   0 ]

LEARNING_RATE_INIT = 1e-3
DECAY_RATE = 0.99
DECAY_STEP = 2000

MINI_BATCH_SIZE = 32
NUM_CHANNELS = 1

ITER_TIMES = 500000
ITER_TIMES_PER_EVALUATE = 20   # 20*32=640
ITER_TIMES_PER_SAVE = 1000
ITER_TIMES_PER_SAVE_MODEL = 10000


TRAIN_SAMPLE_PATH = "D:/QTMT/MSE-CNN_Training_AI3/MSE-CNN_Training_AI/"
VALID_SAMPLE_PATH = "D:/QTMT/MSE-CNN_Training_AI3/MSE-CNN_Training_AI/"

h5f_train = h5py.File(TRAIN_SAMPLE_PATH+'Samples_train.h5', 'r')
h5f_valid = h5py.File(VALID_SAMPLE_PATH+'Samples_valid.h5', 'r')


######################     getfile: 64x64       ###########################

index = 0
size_train = SAMPLE_AMOUNT_LIST[index]
CU_WIDTH = CU_WIDTH_LIST[index]
CU_HEIGHT = CU_HEIGHT_LIST[index]
IMAGES_LENGTH = IMAGES_LENGTH_LIST[index]
SAMPLE_LENGTH = SAMPLE_LENGTH_LIST[index]
LABEL_LENGTH = LABEL_LENGTH_LIST[index]

images_batch, lable_batch, qp_batch,sess, iters_train \
                                    = input_data.get_data_set(h5f_train,index,size_train, CU_WIDTH, CU_HEIGHT, IMAGES_LENGTH, LABEL_LENGTH,MINI_BATCH_SIZE)

######################     net: 64x64       ###########################
#
x = tf.placeholder("float", [None, CU_WIDTH, CU_HEIGHT, NUM_CHANNELS])
y = tf.placeholder("float", [None, LABEL_LENGTH])
qp = tf.placeholder("float", [None, 1])
global_step = tf.placeholder("float")

y_probabilty, y_predict, y_one_hot, total_loss_64x64, accuracy_64x64, learning_rate_current, train_step, opt_vars_all, \
    opt_vars_res1,opt_vars_res2 = net.net_64x64(x, y, qp, global_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP)

######################     feed dict       ###########################

accuracy_64x64_list = []
loss_64x64_list = []

saver = tf.train.Saver(opt_vars_all, write_version=tf.train.SaverDef.V2)
saver_res1 = tf.train.Saver(opt_vars_res1, write_version=tf.train.SaverDef.V2)
saver_res2 = tf.train.Saver(opt_vars_res2, write_version=tf.train.SaverDef.V2)

sess.run(tf.global_variables_initializer())
sess.run(iters_train.initializer)   # 初始化迭代器

for i in range(ITER_TIMES):
    step = i + 1
    feed_step = step

    images_input, lable_input, qp_input =  sess.run([images_batch, lable_batch, qp_batch])

    _, learning_rate, loss, accuracy = sess.run([train_step, learning_rate_current, total_loss_64x64, accuracy_64x64],feed_dict={x: images_input, y: lable_input, qp: qp_input,global_step: feed_step})
    # y, learning_rate , loss, accuracy =sess.run([y_probabilty, learning_rate_current, total_loss_64x64, accuracy_64x64], feed_dict={x:images_input, y: lable_input, qp:qp_input, global_step: feed_step})

    loss_64x64_list.append(loss)
    accuracy_64x64_list.append(accuracy)

    # print("The " + str(i) + " times : loss is " + str(loss) + ",  accuracy is " + str(accuracy) + ", learning_rate is " + str(learning_rate))
    if step % ITER_TIMES_PER_EVALUATE == 0:

        loss = sum(loss_64x64_list[step-ITER_TIMES_PER_EVALUATE:step]) / ITER_TIMES_PER_EVALUATE
        accuracy = sum(accuracy_64x64_list[step-ITER_TIMES_PER_EVALUATE:step]) / ITER_TIMES_PER_EVALUATE
        print("The " + str(i) + " times : loss is " + str(loss) + ",  accuracy is " + str(accuracy) + ", learning_rate is "+str(learning_rate))

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