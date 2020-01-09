
from tensorflow.keras import Input, Model, losses, optimizers, regularizers
from tensorflow.keras.layers import LSTM, Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, GRU, Bidirectional
from tensorflow.keras.layers import Reshape, concatenate, add, PReLU, ZeroPadding2D, Dropout, BatchNormalization


def conv2d(input, kernek_size, strides, output_chanel, padding="same", name=None):
    conv1 = Conv2D(output_chanel, kernek_size, strides, name=name, kernel_regularizer=regularizers.l2(1e-4),
                   padding=padding, activation="relu", use_bias=True)(input)
    # conv2 = Conv2D(output_chanel, kernek_size, strides,
    #                padding = "same", activation = "relu", use_bias = True)(conv1)
    return conv1


def maxPool(input, pool_size, name=None):
    pool = MaxPooling2D(pool_size, padding="same", name=name)(input)
    return pool

def paddingReshape(input, newshape, padding_size):
    return ZeroPadding2D(padding_size)(Reshape(newshape)(input))

def padding1dConv2d(input, kernek_length, strides, inChanell, outChanell, name=None):
    # return Conv3D(padding, (kernek_length, inChanell, 1), (strides[0], strides[1], 1), name=name,
    #                padding="same", activation="relu", use_bias=True)(Reshape((-1, inChanell, 1))(input))
    return conv2d(paddingReshape(input, (-1, inChanell, 1), (kernek_length // 2, 0)),
           (kernek_length, inChanell), strides, outChanell, padding="valid", name=name)

def build_modle(video_frames=120, audio_frames=60, audio_height=64):
    cnum = 128
    video_input = Input(shape=(video_frames, 17 * 4), name='video_input')
    # audio_input = Input(shape=(audio_frames, audio_height), name='audio_input')
    # fc0_video = PReLU()(Dense(64, use_bias=True, bias_initializer='zeros', name="fc0_video")(video_input))

    # conv1a_video = padding1dConv2d(video_input, 7, (1, 1), 17 * 4, cnum, name="conv1a_video")
    # conv1b_video = padding1dConv2d(conv1a_video, 7, (1, 1), cnum, cnum, name="conv1b_video")
    # conv1c_video = padding1dConv2d(conv1b_video, 7, (1, 1), cnum, cnum, name="conv1c_video")
    # add1_video = add([conv1a_video, conv1c_video], name="add1_video")
    #
    # conv2a_video = padding1dConv2d(add1_video, 7, (1, 1), cnum, 2 * cnum, name="conv2a_video")
    # conv2b_video = padding1dConv2d(conv2a_video, 7, (1, 1), 2 * cnum, 2 * cnum, name="conv2b_video")
    # conv2c_video = padding1dConv2d(Dropout(0.9)(conv2b_video), 7, (1, 1), 2 * cnum, 2 * cnum, name="conv2c_video")
    # add2_video = add([conv2a_video, conv2c_video], name="add2_video")
    #
    # conv3a_video = padding1dConv2d(add2_video, 7, (1, 1), 2 * cnum, 4 * cnum, name="conv3a_video")
    # conv3b_video = padding1dConv2d(conv3a_video, 7, (1, 1), 4 * cnum, 4 * cnum, name="conv3b_video")
    # conv3c_video = padding1dConv2d(Dropout(0.9)(conv3b_video), 7, (1, 1), 4 * cnum, 4 * cnum, name="conv3c_video")
    # add3_video = add([conv3a_video, conv3c_video], name="add3_video")
    poll3a_video = maxPool(Reshape((-1, 17 * 4, 1))(video_input), (2, 1), name = "pool3a_video")
    video_frames = video_frames // 2

    # conv4a_video = padding1dConv2d(poll3a_video, 7, (1, 1), 4 * cnum, 4 * cnum, name="conv4a_video")
    # conv4b_video = padding1dConv2d(conv4a_video, 7, (1, 1), 4 * cnum, 4 * cnum, name="conv4b_video")
    # conv4c_video = padding1dConv2d(conv4b_video, 9, (1, 1), 4 * cnum, 4 * cnum, name="conv4c_video")
    # add4_video = add([conv4a_video, conv4c_video], name="add4_video")
    # fc1_video = Dense(1024, use_bias=True, bias_initializer='zeros', name="fc1_video")(Reshape((-1, cnum * 8))(conv4b_video))


    gru1 = Bidirectional(GRU(4 * cnum, return_sequences=True, kernel_initializer="orthogonal", kernel_regularizer=regularizers.l2(1e-2)),
                          name='lstm_gru1')(Reshape((-1, 4 * 17))(poll3a_video))
    # conv5a_gru1_video = padding1dConv2d(gru1, 7, (1, 1), 8 * cnum, 4 * cnum, name="conv5a_gru1_video")
    # conv5b_gru1_video = padding1dConv2d(conv5a_gru1_video, 7, (1, 1), 4 * cnum, 4 * cnum, name="conv5b_gru1_video")
    # conv5c_gru1_video = padding1dConv2d(conv5b_gru1_video, 7, (1, 1), 4 * cnum, 4 * cnum, name="conv5c_gru1_video")
    # conv5d_gru1_video = padding1dConv2d(conv5c_gru1_video, 9, (1, 1), 4 * cnum, 4 * cnum, name="conv5d_gru1_video")
    # add5_video = add([conv5a_gru1_video, conv5c_gru1_video], name="add5_video")
    # poll5a_video = maxPool(Reshape((-1, 4 * cnum, 1))(add5_video), (2, 1), name="pool5a_video")
    # video_frames = video_frames // 2

    # conv6a_gru1_video = padding1dConv2d(add5_video, 9, (1, 1), 4 * cnum, 2 * cnum, name="conv6a_gru1_video")
    # conv6b_gru1_video = padding1dConv2d(Dropout(0.8)(conv6a_gru1_video), 9, (1, 1), 2 * cnum, 2 * cnum, name="conv6b_gru1_video")
    # conv6c_gru1_video = padding1dConv2d(conv6b_gru1_video, 9, (1, 1), 2 * cnum, 2 * cnum, name="conv6c_gru1_video")
    # add6_video = BatchNormalization()(add([conv6a_gru1_video, conv6c_gru1_video], name="add6_video"))
    gru2 = Bidirectional(GRU(4 * cnum, return_sequences=True, kernel_initializer="orthogonal", bias_initializer='zeros')
                         , name='lstm_gru2')(Reshape((-1, 8 * cnum))(gru1))
    # conv7a_gru2_video = padding1dConv2d(gru2, 7, (1, 1), 8 * cnum, 4 * cnum, name="conv7a_gru2_video")
    # conv7b_gru2_video = padding1dConv2d(conv7a_gru2_video, 7, (1, 1), 4 * cnum, 4 * cnum, name="conv7b_gru2_video")
    # conv7c_gru2_video = padding1dConv2d(conv7b_gru2_video, 7, (1, 1), 4 * cnum, 4 * cnum, name="conv7c_gru2_video")
    # add7_video = add([conv7a_gru2_video, conv7c_gru2_video], name="add7_video")

    # gru_output = (concatenate([Reshape((-1, 4 * cnum, 1))(add5_video), Reshape((-1, 4 * cnum, 1))(add7_video)], axis=3))

    # conv8a_video = padding1dConv2d(gru_output, 7, (1, 1), 8 * cnum, 4 * cnum, name="conv8a_output_video")
    # conv8b_video = padding1dConv2d(conv8a_video, 7, (1, 1), 4 * cnum, 4 * cnum, name="conv8b_output_video")
    # conv8c_video = padding1dConv2d(conv8b_video, 7, (1, 1), 4 * cnum, 4 * cnum, name="conv8c_output_video")
    # add8_video = add([conv8a_video, conv8c_video], name="add8_video")
    #
    # conv9a_fc_video = padding1dConv2d(add8_video, 7, (1, 1), 4 * cnum, 2 * cnum, name="conv9a_fc_video")
    # conv9b_fc_video = padding1dConv2d(conv9a_fc_video, 7, (1, 1), 2 * cnum, 2 * cnum, name="conv9b_fc_video")
    # conv9c_fc_video = padding1dConv2d(conv9b_fc_video, 7, (1, 1), 2 * cnum, 2 * cnum, name="conv9c_fc_video")
    # add9_video = add([conv9a_fc_video, conv9c_fc_video], name="add9_video")

    # fc8a_video = Dense(video_frames * cnum, activation="relu", use_bias=True, kernel_regularizer=regularizers.l2(1e-3), name="fc8a_video")\
    #     (Reshape((video_frames * cnum * 2,))(add9_video))
    # fc8b_video = Dense(video_frames * cnum, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fc8b_video") \
    #     (Reshape((video_frames * cnum,))(fc8a_video))
    # fc8c_video = Dense(video_frames * cnum, name="fc8c_video") \
    #     (Reshape((video_frames * cnum,))(fc8b_video))
    # conv8d_video = padding1dConv2d(fc8b_video, 5, (1, 1), cnum , audio_height, name="conv8d_output_video")
    # output = padding1dConv2d(conv8d_video, 5, (1, 1), cnum, audio_height, name="output_video")

    output = Dense(audio_height, name="output_video")(gru2)
    model = Model(inputs=[video_input], outputs=[Reshape((-1, audio_height))(output)])
    model.summary()
    # plot_model(model, to_file='model.png')
    return model

