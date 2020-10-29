import tensorflow as tf
import net_cu as net
import h5py
import warnings
warnings.filterwarnings("ignore")

MINI_BATCH_SIZE = 32
ITER_TIMES = 500
NUM_CHANNELS = 1

CU_WIDTH_LIST = [64, 32, 16, 32, 8, 32, 16, 8, 32, 16]
CU_HEIGHT_LIST = [64, 32, 16, 16, 8, 8, 8, 4, 4, 4]
# IMAGE_SIZE = [[64,64],[32,32],[16,16],[32,16],[8,8],[32,8],[16,8],[8,4],[32,4],[16,4]]
SAMPLE_LENGTH_LIST = [4105, 1033, 265, 521, 73, 265, 137, 41, 137, 73]
IMAGES_LENGTH_LIST = [4096, 1024, 256, 512, 64, 256, 128, 32, 128, 64]

LEARNING_RATE_INIT = 1e-4
DECAY_RATE = 0.01
DECAY_STEP = 2000

ITER_TIMES_PER_EVALUATE = 20
ITER_TIMES_PER_SAVE = 1000
ITER_TIMES_PER_SAVE_MODEL = 10000

TRAINSET_READSIZE = 1000
VALIDSET_READSIZE = 1000
# TRAINSET_READSIZE = 800000
# VALIDSET_READSIZE = 100000

h5f_train = h5py.File('Samples_train.h5', 'r')
h5f_valid = h5py.File('Samples_valid.h5', 'r')

######################     train: 64x64       ###########################
index = 0
CU_WIDTH = CU_WIDTH_LIST[index]
CU_HEIGHT = CU_HEIGHT_LIST[index]
CU_NAME = str(CU_WIDTH)+'x'+str(CU_HEIGHT)
IMAGES_LENGTH = IMAGES_LENGTH_LIST[index]
SAMPLE_LENGTH = SAMPLE_LENGTH_LIST[index]


data_buff_train = h5f_train[CU_NAME]

size_train = len(data_buff_train)

# buf_sample = patch_Y + rdcost + qp + rdcost_min + partition_model
images_train = data_buff_train[0:size_train-1,:IMAGES_LENGTH]
label_train = data_buff_train[0:size_train-1,4104]
qp_train = data_buff_train[0:size_train-1,4102]
assert images_train.shape[0] == label_train.shape[0]
assert images_train.shape[0] == qp_train.shape[0]

sess = tf.Session()
images_input = images_train.reshape([-1,CU_WIDTH,CU_HEIGHT,1])
label_input = tf.one_hot(indices=label_train, depth=2).eval(session=sess)
qp_input = qp_train.reshape([-1, 1])

# 将array转化为tensor
data_sets_train = tf.data.Dataset.from_tensor_slices((images_input, label_input, qp_input))

# 从data数据集中按顺序抽取buffer_size个样本放在buffer中，然后打乱buffer中的样本
# buffer中样本个数不足buffer_size，继续从data数据集中安顺序填充至buffer_size，
# 此时会再次打乱
data_sets_train.shuffle(TRAINSET_READSIZE)
# data_sets_valid.shuffle(VALIDSET_READSIZE)

# 每次从buffer中抽取32个样本
data_sets_train = data_sets_train.batch(MINI_BATCH_SIZE)
# data_sets_valid = data_sets_valid.batch(MINI_BATCH_SIZE)

# 将data数据集重复
data_sets_train = data_sets_train.repeat()
# data_sets_valid = data_sets_valid.repeat()

# 构造获取数据的迭代器
iters_train = data_sets_train.make_initializable_iterator()
# iters_valid = data_sets_valid.make_initializable_iterator()

# 每次从迭代器中获取一批数据
data_train = iters_train.get_next()
# batch_images_valid, batch_label_valid, batch_qp_valid = iters_valid.get_next()

# tf.reset_default_graph()

# sess.run(tf.global_variables_initializer())

accuracy_64x64_list = []
loss_64x64_list = []
# evaluate_loss_accuracy(iter_times_last, LEARNING_RATE_INIT)

# total_loss_64x64, accuracy_64x64, learning_rate_current, train_step, opt_vars_all, opt_vars_res1, opt_vars_res2 \
#     = net.net_64x64(images_input, label_input, qp_input, global_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP)

for i in range(ITER_TIMES):
    step = i + 1

    feed_step = step

    sess.run(tf.global_variables_initializer())
    loss, accuracy, learning_rate_value, train_step = sess.run(net.net_64x64(images_input, label_input, qp_input, feed_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP))
    # train_batch = data_sets_train.batch(MINI_BATCH_SIZE)
    # images_input, label_input, qp_input, rd_input, rdmin_input = data_sets.train.next_batch(MINI_BATCH_SIZE)
    # qp_input = np.reshape(qp_input,[32,1])

    # learning_rate_value, _, loss, accuracy = sess.run([learning_rate_current, train_step, total_loss_64x64, accuracy_64x64], feed_dict={x:images_input, y:label_input.eval(session=sess), qp:qp_input, global_step:feed_step})
    # sess.run([learning_rate_current, train_step], feed_dict={x:batch[0], y:batch[1], qp:batch[2], rd_cost:batch[3], rd_cost_min:batch[4]})

    loss_64x64_list.append(loss)
    accuracy_64x64_list.append(accuracy)

    if step % ITER_TIMES_PER_EVALUATE == 0:
        print("For %d step: loss is %lf, accruacy is %lf, learning_rate is %lf \n"%(i, loss, accuracy, learning_rate_value))

#     if step % ITER_TIMES_PER_SAVE == 0:
#         saver.save(sess, 'Models/ing/model_64x64_%d.dat'%step)
#         saver_res1.save(sess, 'Models/ing/model_res1_%d.dat' % step)
#         saver_res2.save(sess, 'Models/ing/model_res2_%d.dat' % step)
#
# saver.save(sess, 'Models/final/model_64x64.dat')
# saver_res1.save(sess, 'Models/final/model_res1.dat')
# saver_res2.save(sess, 'Models/final/model_res2.dat')