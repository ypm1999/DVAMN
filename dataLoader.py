import json
import numpy as np
from utils import print_log
import pickle, cv2


class dataLoader:
    train_iter = 0
    validation_iter = 0
    joint_relation = [[1, 3], [2, 4], [0, 1], [0, 2], [0, 17], [5, 17], [6, 17], [5, 7], [6, 8], [7, 9], [8, 10],
                      [11, 17], [12, 17], [11, 13], [12, 14], [13, 15], [14, 16]]

    def draw_skeleton(self, pose):
        size = 64
        boxl = np.min(pose, axis=0)
        [width, height] = np.max(pose, axis=0) - boxl
        joints = np.zeros((18, 2), dtype='int32')
        for i, p in enumerate(pose):
            x_ratio = (p[0] - boxl[0]) / width
            y_ratio = (p[1] - boxl[1]) / height
            joints[i][0] = min(size - 1, int(round(x_ratio * size)))
            joints[i][1] = min(size - 1, int(round(y_ratio * size)))
        joints[-1] = [int((joints[5][0] + joints[6][0]) / 2), int((joints[5][1] + joints[6][1]) / 2)]
        color = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        skeleton = np.zeros((size, size, 1), dtype="float32")
        for i in range(len(self.joint_relation)):
            cv2.line(skeleton, tuple(joints[self.joint_relation[i][0]]), tuple(joints[self.joint_relation[i][1]]), (1), 2)
        skeleton = skeleton.reshape(size, size)
        return skeleton

    def redump(self):
        videos_fps = json.load(open("data_double/fps_train_videos.json", 'r'))
        keys = list(videos_fps.keys())
        start = 7
        rang = list(range(start, len(keys), 8))
        for i in rang:
            video = keys[i]
            leng = int(videos_fps[video])
            # print_log(f"running[{i}/{len(keys)}] {video} -> {leng}fps")
            res = {}
            for j in range(1, leng + 2, 3):
                fn = f"{video}_frame_{j}.jpg"
                if not self.pose.__contains__(fn):
                    continue
                res[j] = self.draw_skeleton(self.pose.get(fn).reshape(17, 2))
            print_log(f"saving[{i}/{len(keys)}] {video}: {leng}fps -> {len(res)}fps")
            pickle.dump(res, open(f'data_double/pose_64/{video}_64.pkl', 'wb'))
        # cnt = 0
        # for key, item in self.pose.items():
        #     cnt += 1
        #     if (cnt % 1000 == 0):
        #         print_log(f"[{cnt}/{len(self.pose)}] {key}")
        #     id, i = key[:-4].split("_frame_")
        #     if not self.pose_img_cache.__contains__(id):
        #         self.pose_img_cache[id] = {}
        #     self.pose_img_cache[id][int(i)] = self.draw_skeleton(item.reshape(17, 2))
        #     if cnt % 100 * 1000 == 0:
        #         pickle.dump(self.pose_img_cache, open('data/pose_48_pic.pkl', 'wb'))
        # for key, item in self.pose_img_cache.items():
        #     print_log(f"saving {key}")
        #     pickle.dump(item, open(f'data/pose_48/{key}.pkl', 'wb'))
        # pickle.dump(self.pose_img_cache, open('data/pose_48_pic.pkl', 'wb'))

    def __init__(self):
        train_fragment_json = 'data_double/fragment_train_shuffled.json'
        validate_fragment_json = 'data_double/fragment_validation_shuffled.json'
        video_data_pkl = 'data_double/pose_norm_train_videos.pkl'
        video_res_data_pkl = 'data_double/pose_res_train_videos.pkl'
        audio_data_pkl = 'data/audio_train_10fps_vggish.pkl'
        # train_fragment_json = 'data/fragment_train_shuffled.json'
        # validate_fragment_json = 'data/fragment_validation_shuffled.json'
        # video_data_pkl = 'data/pose_norm_train_videos.pkl'
        # video_res_data_pkl = 'data/pose_res_train_videos.pkl'
        # audio_data_pkl = 'data/audio_train_10fps_vggish.pkl'

        self.train_fragment = json.load(open(train_fragment_json, 'r'))
        self.validate_fragment = json.load(open(validate_fragment_json, 'r'))
        np.random.shuffle(self.train_fragment)
        np.random.shuffle(self.validate_fragment)
        self.train_data_size = len(self.train_fragment)
        self.validation_data_size = len(self.validate_fragment)
        self.pose = pickle.load(open(video_data_pkl, 'rb'))
        self.pose_res = pickle.load(open(video_res_data_pkl, 'rb'))
        self.audio = pickle.load(open(audio_data_pkl, 'rb'))
        self.pose_img_cache = {}
        # videos = json.load(open("data_double/fps_train_videos.json", 'r')).keys()
        # for key in videos:
        #     self.pose_img_cache[key] = pickle.load(open(f"data_double/pose_64/{key}_64.pkl", 'rb'))
        print_log("Load finished")

    def load_video_data_as_image(self, fragment, iter, batch=-1):
        if batch == -1:
            batch = len(fragment)
        data = np.zeros(shape=(batch, 120, 64, 64))
        for i, index in enumerate(range(iter, batch + iter)):
            frag = fragment[index % len(fragment)]
            tmp = self.pose_img_cache[frag[0]]
            for j, frame in enumerate(range(frag[1], frag[2], 3)):
                data[i][j] = tmp[frame]
        return data.reshape((batch, 120, 64, 64, 1))

    def load_video_data(self, fragment, iter, batch=-1):
        if batch == -1:
            batch = len(fragment)
        data = np.zeros(shape=(batch, 120, 34))
        for i, index in enumerate(range(iter, batch + iter)):
            frag = fragment[index % len(fragment)]
            for j, frame in enumerate(range(frag[1], frag[2], 3)):
                fn = "%s_frame_%d.jpg" % (frag[0], frame)
                data[i][j] = self.pose[fn]
        return data

    def load_video_res_data(self, fragment, iter, batch=-1):
        if batch == -1:
            batch = len(fragment)
        data = np.zeros(shape=(batch, 120, 34))
        for i, index in enumerate(range(iter, batch + iter)):
            frag = fragment[index % len(fragment)]
            for j, frame in enumerate(range(frag[1], frag[2], 3)):
                fn = "%s_frame_%d.jpg" % (frag[0], frame)
                data[i][j] = self.pose_res[fn]
        return data

    def load_audio_data(self, fragment, iter, batch):
        if batch == -1:
            batch = len(fragment)

        length = 60
        data = np.zeros(shape=(batch, length, 128))
        for i, index in enumerate(range(iter, batch + iter)):
            frag = fragment[index % len(fragment)]
            l = (frag[1] - 1) // 3 - 5
            r = (frag[2] - 1) // 3 - 5
            tmp = self.audio[frag[0]][l:r:120//length, :]
            if tmp.shape != (length, 128):
                print(f"{frag[0]}[{self.audio[frag[0]].shape}]: {l}->{r}")
            data[i] = tmp
        # data = np.reshape(data, newshape=(batch, 60, 34))
        # data = np.mean(data, axis=2)
        # print(data.shape)
        return data

    def get_train_batch(self, batch_size):
        '''
        :param batch_size:
        :return: [video_data_batch, audio_data_batch]
        video_data_batch = [batch_size] * [frames pre video] * [34-dim data of a frame]
        audio_data_batch = [batch_size] * [frames pre audio] * [34-dim data of a frame]
        NOW frames pre video = 150, frames pre audio = 300
        '''

        video = self.load_video_data(self.train_fragment, self.train_iter, batch_size)
        video_res = self.load_video_res_data(self.train_fragment, self.train_iter, batch_size)
        audio = self.load_audio_data(self.train_fragment, self.train_iter, batch_size)
        video = np.concatenate([video, video_res], axis=2)
        self.train_iter += batch_size
        return (video, audio)

    def get_validation_batch(self, batch_size):
        '''
        :param batch_size:
        :return: [video_data_batch, audio_data_batch]
        video_data_batch = [batch_size] * [frames pre video] * [34-dim data of a frame]
        audio_data_batch = [batch_size] * [frames pre audio] * [34-dim data of a frame]
        NOW frames pre video = 150, frames pre audio = 300
        '''
        video = self.load_video_data(self.validate_fragment, self.validation_iter, batch_size)
        video_res = self.load_video_res_data(self.validate_fragment, self.validation_iter, batch_size)
        audio = self.load_audio_data(self.validate_fragment, self.validation_iter, batch_size)
        video = np.concatenate([video, video_res], axis=2)
        self.validation_iter += batch_size
        return (video, audio)

    def get_test_batch(self, batch_size):
        '''
        :param batch_size:
        :return: [video_data_batch, audio_data_batch]
        video_data_batch = [batch_size] * [frames pre video] * [34-dim data of a frame]
        audio_data_batch = [batch_size] * [frames pre audio] * [34-dim data of a frame]
        NOW frames pre video = 150, frames pre audio = 300
        '''
        video = self.load_video_data(self.validate_fragment, self.validation_iter, batch_size)
        video_res = self.load_video_res_data(self.validate_fragment, self.validation_iter, batch_size)
        audio = self.load_audio_data(self.validate_fragment, self.validation_iter, batch_size)
        video = np.concatenate([video, video_res], axis=2)
        info = [self.validate_fragment[i % len(self.validate_fragment)]
                for i in range(self.validation_iter, self.validation_iter + batch_size)]
        self.validation_iter += batch_size

        return (info, video, audio)





# class dataLoader:
#     train_iter = 0
#     validation_iter = 0
#
#     def __init__(self):
#         train_fragment_json = 'data_old/fragment_train_shuffled.json'
#         validate_fragment_json = 'data_old/fragment_validation_shuffled.json'
#         video_data_npy = 'data_old/pose.npy'
#         audio_data_pkl = 'data_old/audio_extend_train.pkl'
#
#         self.train_fragment = json.load(open(train_fragment_json, 'r'))
#         self.validate_fragment = json.load(open(validate_fragment_json, 'r'))
#         np.random.shuffle(self.validate_fragment)
#         self.train_data_size = len(self.train_fragment)
#         self.validation_data_size = len(self.validate_fragment)
#         self.pose = np.load(video_data_npy, allow_pickle=True).item()
#         self.audio = pickle.load(open(audio_data_pkl, 'rb'))
#         print_log("Load finished")
#         # self.pose = json.load(open(self.video_data_json, 'r'))
#         # for key, value in self.pose.items():
#         #     self.pose[key] = np.reshape(np.array(value), newshape=(34))
#         # np.save("data/pose.npy", self.pose)
#
#     def load_video_data(self, fragment, iter, batch=-1):
#         if batch == -1:
#             batch = len(fragment)
#         data = np.zeros(shape=(batch, 150, 34))
#         for i, index in enumerate(range(iter, batch + iter)):
#             frag = fragment[index % len(fragment)]
#             for j, frame in enumerate(range(frag[1], frag[2], 3)):
#                 fn = "%s_frame_%d.jpg" % (frag[0], frame)
#                 data[i][j] = self.pose[fn]
#         return data
#
#     def load_audio_data(self, fragment, iter, batch):
#         if batch == -1:
#             batch = len(fragment)
#         data = np.zeros(shape=(batch, 150, 128))
#         for i, index in enumerate(range(iter, batch + iter)):
#             frag = fragment[index % len(fragment)]
#             data[i] = self.audio[frag[0]][:, int((frag[1] - 1) / 3):int((frag[2] - 1) / 3)].T
#         data = np.reshape(data, newshape=(batch, 75, -1, 128))
#         data = np.mean(data, axis=2)
#         # print(data.shape)
#         return data
#
#     def get_train_batch(self, batch_size):
#         '''
#         :param batch_size:
#         :return: [video_data_batch, audio_data_batch]
#         video_data_batch = [batch_size] * [frames pre video] * [34-dim data of a frame]
#         audio_data_batch = [batch_size] * [frames pre audio] * [34-dim data of a frame]
#         NOW frames pre video = 150, frames pre audio = 300
#         '''
#         video = self.load_video_data(self.train_fragment, self.train_iter, batch_size)
#         audio = self.load_audio_data(self.train_fragment, self.train_iter, batch_size)
#         self.train_iter += batch_size
#         return (video, audio)
#
#     def get_validation_batch(self, batch_size):
#         '''
#         :param batch_size:
#         :return: [video_data_batch, audio_data_batch]
#         video_data_batch = [batch_size] * [frames pre video] * [34-dim data of a frame]
#         audio_data_batch = [batch_size] * [frames pre audio] * [34-dim data of a frame]
#         NOW frames pre video = 150, frames pre audio = 300
#         '''
#         video = self.load_video_data(self.validate_fragment, self.validation_iter, batch_size)
#         audio = self.load_audio_data(self.validate_fragment, self.validation_iter, batch_size)
#         self.validation_iter += batch_size
#         return (video, audio)
#
#     def get_test_batch(self, batch_size):
#         '''
#         :param batch_size:
#         :return: [video_data_batch, audio_data_batch]
#         video_data_batch = [batch_size] * [frames pre video] * [34-dim data of a frame]
#         audio_data_batch = [batch_size] * [frames pre audio] * [34-dim data of a frame]
#         NOW frames pre video = 150, frames pre audio = 300
#         '''
#         video = self.load_video_data(self.validate_fragment, self.validation_iter, batch_size)
#         audio = self.load_audio_data(self.validate_fragment, self.validation_iter, batch_size)
#         info = [self.validate_fragment[i % len(self.validate_fragment)]
#                 for i in range(self.validation_iter, self.validation_iter + batch_size)]
#         self.validation_iter += batch_size
#
        return (info, video, audio)