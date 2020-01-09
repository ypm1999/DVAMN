
from tensorflow.keras import Input, Model, losses, optimizers
from tensorflow.keras.layers import LSTM, Dense, Conv2D, MaxPool2D, concatenate, Flatten, GRU, Bidirectional, PReLU


def conv2d(input, kernek_size, strides, output_chanel, name=None):
    conv1 = Conv2D(output_chanel, kernek_size, strides, name=name,
                   padding="same", activation="relu", use_bias=True)(input)
    # conv2 = Conv2D(output_chanel, kernek_size, strides,
    #                padding = "same", activation = "relu", use_bias = True)(conv1)
    return conv1


def maxPool(input, pool_size, name=None):
    pool = MaxPool2D(pool_size, padding="same", name=name)(input)
    return pool


def build_modle(video_frames=120, audio_frames=120, audio_height=32):
    video_input = Input(shape=(video_frames, 17 * 2), name='video_input')
    # audio_input = Input(shape=(frame_pre_video, audio_height), name='audio_input')
    fc0_video = PReLU()(Dense(128, use_bias=True, bias_initializer='zeros', name="fc0_video")(video_input))
    fc1_video = PReLU()(Dense(256, use_bias=True, bias_initializer='zeros', name="fc1_video")(fc0_video))
    fc2_video = PReLU()(Dense(512, use_bias=True, bias_initializer='zeros', name="fc2_video")(fc1_video))
    fc3_video = PReLU()(Dense(1024, use_bias=True, bias_initializer='zeros', name="fc3_video")(fc2_video))
    # fc4_video = PReLU()(Dense(1024, use_bias=True, bias_initializer='zeros', name="fc4_video")(fc3_video))
    gru1 = Bidirectional(GRU(512, return_sequences=True, kernel_initializer="orthogonal", bias_initializer='zeros'),
                         name='lstm_gru1')(fc3_video)
    fc1_gru1 = Dense(1024, use_bias=True, bias_initializer='zeros', name="fc1_gru1")((gru1))
    fc2_gru1 = Dense(1024, use_bias=True, bias_initializer='zeros', name="fc2_gru1")(fc1_gru1)

    gru2 = Bidirectional(GRU(1024, return_sequences=True, kernel_initializer="orthogonal", bias_initializer='zeros')
                         , name='lstm_gru2')(fc2_gru1)
    fc1_gru2 = Dense(1024, use_bias=True, bias_initializer='zeros', name="fc1_gru2")(gru2)
    fc2_gru2 = Dense(1024, use_bias=True, bias_initializer='zeros', name="fc2_gru2")(fc1_gru2)
    gru_output = concatenate([fc2_gru1, fc2_gru2], axis=2)

    fc1_output = Dense(512, activation="linear", name="fc1_output")(gru_output)
    fc2_output = Dense(256, activation="linear", name="fc2_output")(fc1_output)
    output = Dense(audio_height, activation="linear", name="output")(fc2_output)

    model = Model(inputs=[video_input], outputs=[output])
    # model.summary()
    # plot_model(model, to_file='model.png')
    return model

