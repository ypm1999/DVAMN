import tensorflow as tf
import numpy as np
import yaml, time, datetime,random, os
from tensorflow import keras
from tensorflow.keras import losses, optimizers, metrics
from tensorflow.keras.utils import plot_model
from tensorflow.contrib.opt import AdamWOptimizer
from dataLoader import dataLoader
from utils import print_log
from modle_conv_GRU import build_modle


from tensorflow.python.client import device_lib
if (not (any((x.device_type == 'GPU') for x in device_lib.list_local_devices()))):
    print_log("Using CPU")
    from tensorflow.core.protobuf import rewriter_config_pb2
    from tensorflow.keras.backend import set_session
    tf.keras.backend.clear_session()  # For easy reset of notebook state.
    config_proto = tf.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.arithmetic_optimization = off
    config_proto.graph_options.rewrite_options.memory_optimization = off
    # config_proto.gpu_options.allow_growth=True
    session = tf.Session(config=config_proto)
    set_session(session)

def MSE(x, y):
    return np.sqrt(np.sum(np.square((x - y))))

def logcosh(x, y):
    x = x - y
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return np.sum(s + np.log1p(p) - np.log(2))


def get_answer(info, _y, y):
    answers = []
    min_val = []
    answers_random = info[:]
    random.shuffle(answers_random)
    for now in _y:
        min_mse = 1e100
        ans = -1
        for j, tmp in enumerate(y):
            mse = logcosh(now, tmp)
            if mse < min_mse:
                min_mse = mse
                ans = j
        answers.append(info[ans])
        min_val.append(min_mse)
    return (answers, min_val, answers_random)

def test(loader, size = 2):
    model = tf.keras.models.load_model("log/20191130-003024/model-snap70000.h5")
    model.compile('adam', 'logcosh', metrics=['mse'])
    total = 0
    ans_total = 0
    ans_random_total = 0
    while(1):
        info, x, y = loader.get_test_batch(size)
        y = np.reshape(y, newshape=(size, -1))
        _y = model.predict_on_batch(x)
        _y = np.reshape(_y, newshape=(size, -1))
        answers ,temp, rand_answers = get_answer(info, _y, y)
        ans = 0
        ans_rand = 0
        for i in range(size):
            if (info[i] == answers[i]):
                ans += 1
            if (info[i] == rand_answers[i]):
                ans_rand += 1
        # for i in range(size):
        #     print("{} -> {} - {} - ({})".format(info[i], answers[i], info[i] == answers[i], temp[i]))
        # print('---------------------------------')
        ans_total += ans
        ans_random_total += ans_rand
        total += size
        print("[{}] acc :{}, total_acc: {}, rand_acc: {}, total_rand_acc: {}".
              format(total, ans / size, ans_total/total, ans_rand/size, ans_random_total/total))


def metrics(loader, size = 128):
    # import tensorflow.keras.backend as K
    #
    # def L1(y_true, y_pred):
    #     return K.sqrt(K.mean(K.square(y_pred - y_true)))

    model = tf.keras.models.load_model("log/20191130-003024/model-snap70000.h5")
    adam = optimizers.Nadam()
    model.compile(adam, 'logcosh', metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    length = 10000
    LOSS = np.zeros(shape=(length))
    RMSE = np.zeros(shape=(length))
    MAE = np.zeros(shape=(length))
    for i in range(length):
        x, y = loader.get_validation_batch(size)
        loss, rmse, mae = model.test_on_batch(x, y)
        LOSS[i] = np.mean(loss)
        RMSE[i] = np.mean(rmse)
        MAE[i] = np.mean(mae)
        print_log(f"running {i}, LOSS = {LOSS[i]}, RMSE = {RMSE[i]}, MAE = {MAE[i]}")
    LOSS = np.mean(LOSS)
    RMSE = np.mean(RMSE)
    MAE = np.mean(MAE)
    print_log(f"LOSS = {LOSS}, RMSE = {RMSE}, MAE = {MAE}")
    return LOSS, RMSE, MAE


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)
    loader = dataLoader()
    # test(loader)
    # metrics(loader)
    # exit(0)
    sess = tf.Session()
    logdir = "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)
    os.system("cp ./*.py " + logdir)
    os.system("cp ./config.yml " + logdir)
    
    

    model = build_modle(120, 60, 128)
    # model = tf.keras.models.load_model("log/20191125-215237/model-snap174000.h5")
    # exit(0)


    plot_model(model, logdir + "model.png")
    with open(logdir + 'modle_summary.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    config = yaml.load(open("config.yml", "r"))

    # sgd = optimizers.SGD(lr=config["learning_rate"], momentum=0.9)
    adam = optimizers.Nadam(lr=config["learning_rate"])
    model.compile(adam, 'logcosh', metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    train_iter = config["max_iteration"]
    batch_size = config["batch_size"]
    save_interval = config["model_save_interval"]
    validation_interval = config["validation_interval"]

    with tf.device('/cpu:0'):
        TRAIN_LOSS = tf.placeholder(tf.float64, [])
        VALIDATION_LOSS = tf.placeholder(tf.float64, [])
        MAE = tf.placeholder(tf.float64, [])
        RMSE = tf.placeholder(tf.float64, [])
        ACC = tf.placeholder(tf.float64, [])
        ACC_T = tf.placeholder(tf.float64, [])
        ACC_RAND = tf.placeholder(tf.float64, [])
        ACC_RAND_T = tf.placeholder(tf.float64, [])
        tf.summary.scalar("LOSS_VALIDATION", VALIDATION_LOSS)
        tf.summary.scalar("LOSS_TRAIN", TRAIN_LOSS)
        tf.summary.scalar("RMSE", RMSE)
        tf.summary.scalar("MAE", MAE)
        tf.summary.scalar("acc", ACC)
        tf.summary.scalar("acc_T", ACC_T)
        tf.summary.scalar("acc_RAND", ACC_RAND)
        tf.summary.scalar("acc_RAND_T", ACC_RAND_T)
        merged_summary = tf.summary.merge_all()


    total = 0
    acc_1000 = []
    ans_random_total = 0
    size = 2
    increase_interval = 16

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    for i in range(train_iter):
        used_time = time.time()
        x, y = loader.get_train_batch(batch_size)
        time1 = time.time() - used_time
        train_out = model.train_on_batch(x, y)
        used_time = time.time() - used_time
        print_log("step[{}]: train_loss={:.3f}, time1={:.3f}, time:{:.3f}".
                  format(i, train_out[0], time1, used_time))

        if (i % save_interval == 0 and i > 0):
            path = "{}model-snap{}.h5".format(logdir, i)
            model.save(path)

        if (i % validation_interval == 0 and i > 0):
            start = time.time()
            x, y = loader.get_train_batch(batch_size * 2)
            val_out = model.test_on_batch(x, y)
            time1 = time.time() - start

            info, x, y = loader.get_test_batch(size)
            y = np.reshape(y, newshape=(size, -1))
            _y = model.predict_on_batch(x)
            _y = np.reshape(_y, newshape=(size, -1))
            answers, temp, rand_answers = get_answer(info, _y, y)
            ans = 0
            ans_rand = 0
            for j in range(size):
                if (info[j] == answers[j]):
                    ans += 1
                if (info[j] == rand_answers[j]):
                    ans_rand += 1
            acc_1000.append(ans / size)
            if (len(acc_1000) * validation_interval > save_interval * 2):
                acc_1000.pop(0)
            ans_random_total += ans_rand
            total += size
            acc_avg = np.mean(np.array(acc_1000))
            lr = keras.backend.get_value(model.optimizer.lr)
            # if (i % (10000) == 0 and lr > 5e-5):
            #     if (i % (10000 * increase_interval) == 0):
            #         keras.backend.set_value(model.optimizer.lr, lr * increase_interval * 0.6)
            #         increase_interval *= 2
            #     else:
            #         keras.backend.set_value(model.optimizer.lr, lr * 0.8)

            print_log(
                "step[{}]: val_loss={:.3f}, acc={:.3f}, rand_acc={:.3f},\n\t \
                 total_acc={:.5f},  rand_acc_total={:.5f}, lr={:.3}".
                format(i, val_out[0], ans / size, ans_rand / size, acc_avg,
                           ans_random_total / total, lr))

            with tf.device('/cpu:0'):
                summary = sess.run(merged_summary, feed_dict={TRAIN_LOSS: train_out[0], VALIDATION_LOSS: val_out[0],
                                                               RMSE: val_out[1], MAE:val_out[2],
                                                              ACC: ans / size, ACC_T:  acc_avg,
                                                              ACC_RAND: ans_rand / size, ACC_RAND_T: ans_random_total / total
                                                              })
                writer.add_summary(summary, i)
