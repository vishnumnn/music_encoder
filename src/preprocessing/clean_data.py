import librosa
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


from src.preprocessing.models.dbcqt_group import DBCQTGroup

SAMPLE_RATE = 22050
COL_COUNT = 5160
DURATION = 2
FREQ = 84
DB_SCALE_CONST = 4

def mp3s_in_dir(raw_dir):
    paths = []
    for path in glob.glob(os.path.join(raw_dir, "*.mp3")):
        paths.append(path)
    return paths

def ret_ups(running_sps, current_step_count, song_length_s):
    steps_per_s = current_step_count / song_length_s
    if running_sps is None or running_sps == steps_per_s:
        return running_sps
    print(f"Current steps per second {steps_per_s} != running steps per second {running_sps}")
    raise

def power_cqt_from_path(path: str):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=None)
    cqt_amp = np.abs(librosa.cqt(y, sr=sr, n_bins=FREQ))
    return cqt_amp

def dbcqts_from_paths(paths: list[str]) -> list[np.ndarray]:
    dbcqts = []
    running_sps = None
    for path in paths:
        print(f"============ Processing {path} ============")
        song_length_s = int(librosa.get_duration(path=path))
        cqt = power_cqt_from_path(path=path)

        print("Song Duration: ", song_length_s, "CQT Shape: ", cqt.shape)
        running_sps = ret_ups(running_sps=running_sps, current_step_count=cqt.shape[1], song_length_s=song_length_s)
        dbcqt = librosa.amplitude_to_db(cqt, ref=DB_SCALE_CONST)
        print("=========================")
        dbcqts.append(dbcqt)
        break
    return dbcqts

def steps_per_cqt(dbcqts: list[np.ndarray]):
    return [
        dbcqt.shape[1] for dbcqt in dbcqts
    ]

def standardized_dataset(dbcqts: list[np.ndarray]):
    print("========= Standardizing DB CQT Dataset =========")
    steps_for_cqt: list[int] = steps_per_cqt(dbcqts)
    freq_buckets = dbcqts[0].shape[0]
    combined_dataset = np.zeros(shape=(freq_buckets, sum(steps_for_cqt)))
    most_recent_col_filled_idx = 0
    for dbcqt in dbcqts:
        rows, cols = dbcqt.shape
        term_col = most_recent_col_filled_idx + cols
        print(f"Filling combined data set between 0:{rows} and {most_recent_col_filled_idx}:{term_col}")
        combined_dataset[0:rows, most_recent_col_filled_idx:term_col] = dbcqt
        most_recent_col_filled_idx += cols
    print(f"Finished filling combined dataset")
    print(f"Transposing given array of shape: {combined_dataset.shape}")
    transposed_view = np.transpose(combined_dataset)
    print(f"Standardizing within each of {transposed_view.shape[1]} features (frequency buckets) for {transposed_view.shape[0]} samples")
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(transposed_view)
    print(f"Returning standardized combined dataset, shape: {transposed_view.shape}")
    return standardized_data
    

def histogram_for_freq(samples):
    plt.hist(samples, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('amplitude')
    plt.ylabel('frequency')
    plt.title('Histogram of Amplitudes')
    plt.show()

def save_data(dest_file: str, core_dataset: np.ndarray, samples_for_each_song: list[int], steps_per_second: int):
    np.savez_compressed(
        dest_file,
        core_dataset=core_dataset,
        steps_persamples_for_each_song_song=np.array(samples_for_each_song),
        steps_per_second=np.array([steps_per_second])
    )

def clean_and_store_data(dest_file_path, raw_data_dir):
    paths: list[str] = mp3s_in_dir(raw_data_dir)
    dbcqts = dbcqts_from_paths(paths=paths)
    samples_for_each_song = steps_per_cqt(dbcqts)
    dataset = standardized_dataset(dbcqts=dbcqts)
    steps_per_second = dataset.shape[0]
    save_data(
        dest_file=dest_file_path,
        core_dataset=dataset,
        samples_for_each_song=samples_for_each_song,
        steps_per_second=steps_per_second
    )

# def clean_data():
#     paths: list[str] = mp3s_in_dir(DIR)
#     dbcqts = dbcqts_from_paths(paths=paths)
#     samples = np.array([])
#     for db_cqt in dbcqts:
#         for cqt in db_cqt.cqts:
#             samples = np.append(samples, cqt[0, :])
#     db = librosa.amplitude_to_db(samples, ref=np.min)
#     samples2D = samples.reshape(-1, 1)
#     scaler = StandardScaler()
#     standardized_data = scaler.fit_transform(samples2D)
#     x = standardized_data.reshape(1, -1)[0]
#     import pdb; pdb.set_trace()
#     histogram_for_freq(db)