import numpy as np
import json

import pickle

def ref1():
    fs = json.load(open('./audio/30fps_train_videos.json', 'r'))
    # less120 = json.load(open("../less120_train_videos.json"))
    # less140 = json.load(open("../less140_train_videos.json"))
    # less160 = json.load(open("../less160_train_videos.json"))
    data = {}
    fa = []
    for f in fs:
        audio = np.load('./audio/vggish_output/' + f + '.npz')
        print(audio["feature"].shape)
        data[f] = audio["feature"]
        # try:
        #     audio = np.load('./audio/train_feature_0/' + f + '.npz')
        #     # features = ['chroma_stft', 'rms', 'spec_cent', "spec_bw", "rolloff", "zcr", 'melspec']
        #     # features = ['logmelspec']
        #     features = ['logmelspec']
        #     d = []
        #     fal = False
        #     for fea in features:
        #         d.append(audio[fea])
        # except:
        #     print(f)
        #     fal = True
        # if fal:
        #     fa.append(f)
        #     continue
        # d = np.concatenate(d, axis=0).T
        # if not d.shape[1] == 128:
        #     print(f, 'gg')
        # d = np.concatenate([np.zeros((1, 128)), d], axis=0)
        # data[f] = d
    print(len(fs) - len(fa))
    print(len(fa))
    pickle.dump(data, open('data/audio_train_10fps_vggish.pkl', 'wb'))

ref1()
