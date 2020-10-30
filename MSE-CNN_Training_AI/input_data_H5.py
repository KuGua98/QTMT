import tensorflow as tf

TRAINSET_READSIZE = 100 # shuffle size
VALIDSET_READSIZE = 100

def get_data_set(h5file, size_train, size_valid, CU_WIDTH, CU_HEIGHT, IMAGES_LENGTH, LABEL_LENGTH, MINI_BATCH_SIZE):

    ######################     h5文件读取为dataset       ###########################
    # buf_sample = patch_Y + rdcost + qp + rdcost_min + partition_model
    CU_NAME = str(CU_WIDTH) + 'x' + str(CU_HEIGHT)
    data_buff = h5file[CU_NAME]

    ####train dataset####
    images_train = data_buff[0:size_train - 1, :IMAGES_LENGTH]
    label_train = data_buff[0:size_train - 1, 4104]
    qp_train = data_buff[0:size_train - 1, 4102]
    assert images_train.shape[0] == label_train.shape[0]
    assert images_train.shape[0] == qp_train.shape[0]

    sess = tf.Session()
    images_train = images_train.reshape([-1, CU_WIDTH, CU_HEIGHT, 1])
    label_train = tf.one_hot(indices=label_train, depth=LABEL_LENGTH).eval(session=sess)
    qp_train = qp_train.reshape([-1, 1])

    ####valid dataset####
    images_valid = data_buff[0:size_valid - 1, :IMAGES_LENGTH]
    label_valid = data_buff[0:size_valid - 1, 4104]
    qp_valid = data_buff[0:size_valid - 1, 4102]
    assert images_valid.shape[0] == label_valid.shape[0]
    assert images_valid.shape[0] == qp_valid.shape[0]

    sess = tf.Session()
    images_valid = images_valid.reshape([-1, CU_WIDTH, CU_HEIGHT, 1])
    label_valid = tf.one_hot(indices=label_valid, depth=LABEL_LENGTH).eval(session=sess)
    qp_valid = qp_valid.reshape([-1, 1])

    ######################     tensorflow: shuffle/batch       ###########################

    # 将array转化为tensor
    data_sets_train = tf.data.Dataset.from_tensor_slices((images_train, label_train, qp_train))
    data_sets_valid = tf.data.Dataset.from_tensor_slices((images_valid, label_valid, qp_valid))

    # 从data数据集中按顺序抽取buffer_size个样本放在buffer中，然后打乱buffer中的样本
    # buffer中样本个数不足buffer_size，继续从data数据集中安顺序填充至buffer_size，
    # 此时会再次打乱
    data_sets_train = data_sets_train.shuffle(TRAINSET_READSIZE)
    data_sets_valid = data_sets_valid.shuffle(VALIDSET_READSIZE)

    # 每次从buffer中抽取32个样本
    data_sets_train = data_sets_train.batch(MINI_BATCH_SIZE)
    data_sets_valid = data_sets_valid.batch(MINI_BATCH_SIZE)

    # 将data数据集重复
    data_sets_train = data_sets_train.repeat()
    data_sets_valid = data_sets_valid.repeat()

    # 构造获取数据的迭代器
    iters_train = data_sets_train.make_initializable_iterator()
    iters_valid = data_sets_valid.make_initializable_iterator()

    # 每次从迭代器中获取一批数据
    images_batch_train, lable_batch_train, qp_batch_train = iters_train.get_next()
    images_batch_valid, label_batch_valid, qp_batch_valid = iters_valid.get_next()

    return images_batch_train, lable_batch_train, qp_batch_train, sess, iters_train, images_batch_valid, label_batch_valid, qp_batch_valid, iters_valid

