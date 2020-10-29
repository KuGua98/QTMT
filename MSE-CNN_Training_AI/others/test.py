import tensorflow as tf
import net_cu as net
import h5py
import warnings
warnings.filterwarnings("ignore")

MINI_BATCH_SIZE = 32
ITER_TIMES = 500000
NUM_CHANNELS = 1


CU_WIDTH_LIST = [64, 32, 16, 32, 8, 32, 16, 8, 32, 16]
CU_HEIGHT_LIST = [64, 32, 16, 16, 8, 8, 8, 4, 4, 4]
# IMAGE_SIZE = [[64,64],[32,32],[16,16],[32,16],[8,8],[32,8],[16,8],[8,4],[32,4],[16,4]]
LABEL_LENGTH_LIST = [2, 6, 6, 6, 6, 6, 6, 6, 6, 6]
SAMPLE_LENGTH_LIST = [4105, 1033, 265, 521, 73, 265, 137, 41, 137, 73]
IMAGES_LENGTH_LIST = [4096, 1024, 256, 512, 64, 256, 128, 32, 128, 64]

SAMPLE_AMOUNT_LIST = [600]

LEARNING_RATE_INIT = 1e-4
DECAY_RATE = 0.99
DECAY_STEP = 2000

ITER_TIMES_PER_EVALUATE = 20   # 20*32=640
ITER_TIMES_PER_SAVE = 1000
ITER_TIMES_PER_SAVE_MODEL = 10000

TRAINSET_READSIZE = 100 # shuffle size
VALIDSET_READSIZE = 100

TRAIN_SAMPLE_PATH = "E:/QTMT/train_samples/"
VALID_SAMPLE_PATH = "E:/QTMT/valid_samples/"

h5f_train = h5py.File(TRAIN_SAMPLE_PATH+'Samples_train.h5', 'r')
h5f_valid = h5py.File(VALID_SAMPLE_PATH+'Samples_valid.h5', 'r')

adjust_scalar_64 = [-0.3, -0.3,]      # α=0.3  要对应数量以array形式输入
adjust_scalar_else = [-0.3, -0.3, -0.3, -0.3, -0.3, -0.3]

p_64x64 = [  0.25, 0.75]
p_32x32 = []
p_16x16 = []
p_32x16 = []
p_8x8 = []
p_32x8 = []
p_16x8 = []
p_8x4 = []
p_32x4 = []
p_16x4 = []

######################     train: 64x64       ###########################

index = 0
size_train = SAMPLE_AMOUNT_LIST[index]
CU_WIDTH = CU_WIDTH_LIST[index]
CU_HEIGHT = CU_HEIGHT_LIST[index]
CU_NAME = str(CU_WIDTH)+'x'+str(CU_HEIGHT)
IMAGES_LENGTH = IMAGES_LENGTH_LIST[index]
SAMPLE_LENGTH = SAMPLE_LENGTH_LIST[index]
LABEL_LENGTH = LABEL_LENGTH_LIST[index]



######################     h5文件读取为dataset       ###########################

data_buff_train = h5f_train[CU_NAME]
# size_train = len(data_buff_train)
# buf_sample = patch_Y + rdcost + qp + rdcost_min + partition_model
images_train = data_buff_train[0:size_train-1,:IMAGES_LENGTH]
label_train = data_buff_train[0:size_train-1,4104]
qp_train = data_buff_train[0:size_train-1,4102]
assert images_train.shape[0] == label_train.shape[0]
assert images_train.shape[0] == qp_train.shape[0]

sess = tf.Session()
images_train = images_train.reshape([-1,CU_WIDTH,CU_HEIGHT,1])
label_train = tf.one_hot(indices=label_train, depth=LABEL_LENGTH).eval(session=sess)
qp_train = qp_train.reshape([-1, 1])


x = tf.placeholder("float", [None, CU_WIDTH, CU_HEIGHT, NUM_CHANNELS])
y = tf.placeholder("float", [None, LABEL_LENGTH])
qp = tf.placeholder("float", [None, 1])
global_step = tf.placeholder("float")

x = tf.cast(x, tf.float32)
x = tf.scalar_mul(1.0 / 255.0, x)
x_image = tf.reshape(x, [-1, CU_WIDTH, CU_HEIGHT, 1])
y_image = tf.reshape(y, [-1, LABEL_LENGTH])



######################     tensorflow: shuffle/batch       ###########################

# 将array转化为tensor
data_sets_train = tf.data.Dataset.from_tensor_slices((x, y, qp))

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
images_batch, lable_batch, qp_batch = iters_train.get_next()
# batch_images_valid, batch_label_valid, batch_qp_valid = iters_valid.get_next()


total_loss_64x64, accuracy_64x64, learning_rate_current, train_step, opt_vars_all, opt_vars_res1, opt_vars_res2 \
    = net.net_64x64(images_batch, lable_batch, qp_batch, global_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP)

######################     net: 64x64       ###########################


# h_cov = sub.overlap_conv(x_image, 3, 3, 1, 16)
# h_condc = res.condc_lumin_64(h_cov)
# y_probabilty = sub.sub_net_64(h_condc, qp)  # 预测的概率值
#
# y_predict = tf.argmax(y_probabilty, axis=1)  # 返回最大值索引
# y_one_hot = tf.one_hot(indices=y_predict, depth=LABEL_LENGTH)  # 转换为one—hot vector
#
# loss_64_ce = -tf.reduce_sum(tf.multiply(np.power(p_64x64, adjust_scalar_64).astype(np.float32),tf.multiply(y_image, tf.log(y_probabilty + 1e-12)))) / np.sum(np.power(p_64x64, adjust_scalar_64))
# # loss_64_rd = tf.reduce_sum(-tf.multiply(y_image, tf.log()))
# # total_loss_64x64 = loss_64_ce + positive_scalar*loss_64_rd
#
# total_loss_64x64 = loss_64_ce
# accuracy_64x64 = tf.reduce_sum(tf.multiply(y_image, y_one_hot)) / tf.reduce_sum(y_image)
#
# learning_rate_current = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step, DECAY_STEP, DECAY_RATE,staircase=True)
# train_step = tf.train.AdamOptimizer(learning_rate_current).minimize(total_loss_64x64)
#
# opt_vars_all = [v for v in tf.trainable_variables()]
# opt_vars_res1 = tf.trainable_variables(scope='res_unit1')
# opt_vars_res2 = tf.trainable_variables(scope='res_unit2')

######################     feed dict       ###########################

accuracy_64x64_list = []
loss_64x64_list = []

# saver = tf.train.Saver(opt_vars_all, write_version=tf.train.SaverDef.V2)
# saver_res1 = tf.train.Saver(opt_vars_res1, write_version=tf.train.SaverDef.V2)
# saver_res2 = tf.train.Saver(opt_vars_res2, write_version=tf.train.SaverDef.V2)


sess.run(tf.global_variables_initializer())
sess.run(iters_train.initializer,feed_dict={x:images_train, y:label_train, qp:qp_train})   # 初始化迭代器

for i in range(ITER_TIMES):
    step = i + 1
    feed_step = step

    # images_input, lable_input, qp_input =  sess.run([images_batch, lable_batch, qp_batch])

    learning_rate , loss, accuracy =sess.run([learning_rate_current, total_loss_64x64, accuracy_64x64], feed_dict={global_step: feed_step})
    #
    # print(x[0][0])
    # print(y[0][0])
    # print(qp[0])


    loss_64x64_list.append(loss)
    accuracy_64x64_list.append(accuracy)


    if step % ITER_TIMES_PER_EVALUATE == 0:

        loss = sum(loss_64x64_list[step-ITER_TIMES_PER_EVALUATE:step]) / ITER_TIMES_PER_EVALUATE
        accuracy = sum(accuracy_64x64_list[step-ITER_TIMES_PER_EVALUATE:step]) / ITER_TIMES_PER_EVALUATE
        print("The " + str(i) + " times : loss is " + str(loss) + ",  accuracy is " + str(accuracy) + ", learning_rate is "+str(learning_rate))

#     if step % ITER_TIMES_PER_SAVE == 0:
#         saver.save(sess, 'Models/ing/model_64x64_%d.dat'%step)
#         saver_res1.save(sess, 'Models/ing/model_res1_%d.dat' % step)
#         saver_res2.save(sess, 'Models/ing/model_res2_%d.dat' % step)
#
# saver.save(sess, 'Models/final/model_64x64.dat')
# saver_res1.save(sess, 'Models/final/model_res1.dat')
# saver_res2.save(sess, 'Models/final/model_res2.dat')