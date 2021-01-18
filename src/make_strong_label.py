import os
import glob
import numpy as np
import pandas as pd

def v1():
    threshold = 0.99
    audio_time = 60
    input_dir = "../output/sed_result/0110_121231/"
    output_path = "../input/rfcx-species-audio-detection/train_re.csv"
    estimated_event_list = []
    for npy_file in glob.glob(input_dir + '*.npy'):
        framewise_output = np.load(npy_file)
        filename = os.path.split(npy_file)[1]
        recording_id = filename[:9]
        prelabeled_species_id = int(filename[10:-4])
        frame_length = len(framewise_output)

        # 設定したし閾値より高い予測値のもののみでラベル付け
        thresholded = (framewise_output > threshold) * 1
        # 全てのクラスを順に見ていく
        for species_id in range(thresholded.shape[1]):
            # 該当クラスが検知されていない or 該当クラスと事前にラベル付けされているクラスが異なる場合はpass
            if (thresholded[:, species_id].mean() == 0) or (species_id != prelabeled_species_id):
                pass
            else:
                detected = np.argwhere(thresholded[:, species_id]).reshape(-1)  # 全てのframeから検知されているframeのindexのみを取り出す
                head_idx = 0
                tail_idx = 0
                while True:
                    # 音声frameが一つのみ or 音声frameが一つ先のframeと途切れている場合
                    if (tail_idx + 1 == len(detected)) or (detected[tail_idx + 1] - detected[tail_idx] != 1):
                        t_min = detected[head_idx] * (audio_time/frame_length)  # frame -> audioになるようにスケールを合わせる
                        t_max = detected[tail_idx] * (audio_time/frame_length)  # frame -> audioになるようにスケールを合わせる
                        estimated_event = {
                            "recording_id": recording_id,
                            "species_id": species_id,
                            "t_min": t_min.astype(np.float16),
                            "t_max": t_max.astype(np.float16),
                        }
                        estimated_event_list.append(estimated_event)
                        head_idx = tail_idx + 1
                        tail_idx = tail_idx + 1
                        if head_idx >= len(detected):
                            break
                    else:
                        tail_idx += 1

    relabeled_df = pd.DataFrame(estimated_event_list)
    relabeled_df.to_csv(output_path, index=False)


def v2():
    threshold = 0.99
    audio_time = 60
    input_dir = "../output/sed_result/0110_121231/"
    output_path = "../input/rfcx-species-audio-detection/train_re_v2.csv"
    estimated_event_list = []
    for npy_file in glob.glob(input_dir + '*.npy'):
        framewise_output = np.load(npy_file)
        filename = os.path.split(npy_file)[1]
        recording_id = filename[:9]
        prelabeled_species_id = int(filename[10:-4])
        frame_length = len(framewise_output)

        # 設定したし閾値より高い予測値のもののみでラベル付け
        thresholded = (framewise_output > threshold) * 1
        species_ranking = np.argsort(thresholded.sum(axis=0))[::-1]
        # ranking1位がprelabelの種なら2位をtargetにする
        if species_ranking[0] == prelabeled_species_id:
            species_id = species_ranking[1]
        else:
            species_id = species_ranking[0]

        if (thresholded[:, species_id].mean() == 0):
            pass
        else:
            detected = np.argwhere(thresholded[:, species_id]).reshape(-1)  # 全てのframeから検知されているframeのindexのみを取り出す
            head_idx = 0
            tail_idx = 0
            while True:
                # 音声frameが一つのみ or 音声frameが一つ先のframeと途切れている場合
                if (tail_idx + 1 == len(detected)) or (detected[tail_idx + 1] - detected[tail_idx] != 1):
                    t_min = detected[head_idx] * (audio_time/frame_length)  # frame -> audioになるようにスケールを合わせる
                    t_max = detected[tail_idx] * (audio_time/frame_length)  # frame -> audioになるようにスケールを合わせる
                    estimated_event = {
                        "recording_id": recording_id,
                        "species_id": species_id,
                        "t_min": t_min.astype(np.float16),
                        "t_max": t_max.astype(np.float16),
                    }
                    estimated_event_list.append(estimated_event)
                    head_idx = tail_idx + 1
                    tail_idx = tail_idx + 1
                    if head_idx >= len(detected):
                        break
                else:
                    tail_idx += 1

    relabeled_df = pd.DataFrame(estimated_event_list)
    relabeled_df.to_csv(output_path, index=False)


# 間違えているデータのみにそのクラスを学習させる
def v3():
    threshold = 0.95
    audio_time = 60
    input_dir = "../output/sed_result/0110_121231/"
    output_path = "../input/rfcx-species-audio-detection/train_re_v3.csv"
    estimated_event_list = []
    bad_recording_ids = []
    for npy_file in glob.glob(input_dir + '*.npy'):
        framewise_output = np.load(npy_file)
        filename = os.path.split(npy_file)[1]
        recording_id = filename[:9]
        prelabeled_species_id = int(filename[10:-4])
        frame_length = len(framewise_output)

        # 設定したし閾値より高い予測値のもののみでラベル付け
        thresholded = (framewise_output > threshold) * 1
        species_ranking = np.argsort(thresholded.sum(axis=0))[::-1]
        top5_species_id = species_ranking[:5]
        if prelabeled_species_id in top5_species_id:
            pass

        species_id = prelabeled_species_id
        if (thresholded[:, species_id].mean() == 0):
            print(recording_id)  # 正解のクラスを全く予測できていないrecording_idを表示
            bad_recording_ids.append(recording_id)
            pass
        else:
            detected = np.argwhere(thresholded[:, species_id]).reshape(-1)  # 全てのframeから検知されているframeのindexのみを取り出す
            head_idx = 0
            tail_idx = 0
            while True:
                # 音声frameが一つのみ or 音声frameが一つ先のframeと途切れている場合
                if (tail_idx + 1 == len(detected)) or (detected[tail_idx + 1] - detected[tail_idx] != 1):
                    t_min = detected[head_idx] * (audio_time/frame_length)  # frame -> audioになるようにスケールを合わせる
                    t_max = detected[tail_idx] * (audio_time/frame_length)  # frame -> audioになるようにスケールを合わせる
                    estimated_event = {
                        "recording_id": recording_id,
                        "species_id": species_id,
                        "t_min": t_min.astype(np.float16),
                        "t_max": t_max.astype(np.float16),
                    }
                    estimated_event_list.append(estimated_event)
                    head_idx = tail_idx + 1
                    tail_idx = tail_idx + 1
                    if head_idx >= len(detected):
                        break
                else:
                    tail_idx += 1
    bad_recording_ids = "\n".join(bad_recording_ids)
    with open(input_dir + 'bad_recording_ids.txt', 'w') as f:
        f.writelines(bad_recording_ids)

    relabeled_df = pd.DataFrame(estimated_event_list)
    relabeled_df.to_csv(output_path, index=False)


# 正解しているデータのみにそのクラスを学習させる
def v4():
    threshold = 0.99
    audio_time = 60
    input_dir = "../output/sed_result/0110_121231/"
    output_path = "../input/rfcx-species-audio-detection/train_re_v4.csv"
    estimated_event_list = []
    for npy_file in glob.glob(input_dir + '*.npy'):
        framewise_output = np.load(npy_file)
        filename = os.path.split(npy_file)[1]
        recording_id = filename[:9]
        prelabeled_species_id = int(filename[10:-4])
        frame_length = len(framewise_output)

        # 設定したし閾値より高い予測値のもののみでラベル付け
        thresholded = (framewise_output > threshold) * 1
        species_ranking = np.argsort(thresholded.sum(axis=0))[::-1]
        top3_species_id = species_ranking[:3]

        # 上位３つに入っていなければ使わない
        if prelabeled_species_id not in top3_species_id:
            pass

        species_id = prelabeled_species_id
        if (thresholded[:, species_id].mean() == 0):
            print(recording_id)  # 正解のクラスを全く予測できていないrecording_idを表示
            pass
        else:
            detected = np.argwhere(thresholded[:, species_id]).reshape(-1)  # 全てのframeから検知されているframeのindexのみを取り出す
            head_idx = 0
            tail_idx = 0
            while True:
                # 音声frameが一つのみ or 音声frameが一つ先のframeと途切れている場合
                if (tail_idx + 1 == len(detected)) or (detected[tail_idx + 1] - detected[tail_idx] != 1):
                    t_min = detected[head_idx] * (audio_time/frame_length)  # frame -> audioになるようにスケールを合わせる
                    t_max = detected[tail_idx] * (audio_time/frame_length)  # frame -> audioになるようにスケールを合わせる
                    estimated_event = {
                        "recording_id": recording_id,
                        "species_id": species_id,
                        "t_min": t_min.astype(np.float16),
                        "t_max": t_max.astype(np.float16),
                    }
                    estimated_event_list.append(estimated_event)
                    head_idx = tail_idx + 1
                    tail_idx = tail_idx + 1
                    if head_idx >= len(detected):
                        break
                else:
                    tail_idx += 1


    relabeled_df = pd.DataFrame(estimated_event_list)
    relabeled_df.to_csv(output_path, index=False)

if __name__=="__main__":
    v4()