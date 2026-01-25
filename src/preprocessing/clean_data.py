import librosa
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


SAMPLE_RATE = 22050
COL_COUNT = 5160
DURATION = 2
FREQ = 42
_EXPECTED_STEPS_PER_DURATION = 43
INPUT_DIM = DURATION * FREQ * _EXPECTED_STEPS_PER_DURATION
DB_SCALE_CONST = 2

def mp3s_in_dir(raw_dir):
    paths = []
    for path in glob.glob(os.path.join(raw_dir, "*.mp3")):
        paths.append(path)
    return paths

def ret_ups(running_sps, steps_per_s):
    if (running_sps is None or running_sps == steps_per_s) and (steps_per_s == _EXPECTED_STEPS_PER_DURATION):
        return running_sps
    print(f"Current steps per second {steps_per_s} != running steps per second {running_sps}")
    raise

def steps_per_second(step_count, song_length_s):
    return step_count / song_length_s

def steps_per_duration(steps_per_s, duration):
    return int(steps_per_s * duration)

def cutoff_idx_for_cqt(step_count, steps_per_s, duration):
    steps_per_d = steps_per_duration(steps_per_s, duration)
    return (step_count // steps_per_d) * steps_per_d

def power_cqt_from_path(path: str):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=None)
    cqt_amp = np.abs(librosa.cqt(y, sr=sr, n_bins=FREQ))
    return cqt_amp

def dbcqts_from_paths(paths: list[str]) -> list[np.ndarray]:
    dbcqts: list[np.ndarray] = []
    running_sps = None
    steps_per_d = None
    print(f"============ Creating DB CQTs from Raw MP3 data ============")
    for i, path in enumerate(paths):
        print(f"({i})", f"Processing song at {path}")
        song_length_s = int(librosa.get_duration(path=path))
        cqt = power_cqt_from_path(path=path)
        print("Song Duration: ", song_length_s, "CQT Shape: ", cqt.shape)
        steps_per_s = steps_per_second(cqt.shape[1], song_length_s)
        steps_per_d = steps_per_duration(steps_per_s, DURATION)
        running_sps = ret_ups(running_sps=running_sps, steps_per_s=steps_per_s)
        dbcqt = librosa.amplitude_to_db(cqt, ref=DB_SCALE_CONST)
        cutoff = cutoff_idx_for_cqt(step_count=dbcqt.shape[1], steps_per_s=steps_per_s, duration=DURATION)
        dbcqts.append(dbcqt[:, 0:cutoff])
        print("Resizing CQT based on steps_per_second")
        print("steps_per_second:", steps_per_s)
        print("steps_per_duration:", steps_per_d)
        print("New CQT Shape:", dbcqts[-1].shape)
        print("--------------------------------------")
        break
    print(f"============ Finished creating DB CQTs from Raw MP3 Data ============")
    return dbcqts, steps_per_d

def steps_per_cqt(dbcqts: list[np.ndarray]):
    return [
        dbcqt.shape[1] for dbcqt in dbcqts
    ]

def standardized_dataset(dbcqts: list[np.ndarray]):
    print("============ Standardizing DB CQT Dataset ============")
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
    print("============ Finished standardizing DB CQT Dataset ============")
    return standardized_data
    

def histogram_for_freq(samples):
    plt.hist(samples, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('amplitude')
    plt.ylabel('frequency')
    plt.title('Histogram of Amplitudes')
    plt.show()

def save_data(
    dest_file: str,
    core_dataset: np.ndarray,
    samples_for_each_song: list[int],
    steps_per_second: int,
    freq_buckets: int,
):
    print(f"Saving data to {dest_file}.npz ...")
    np.savez_compressed(
        dest_file,
        core_dataset=core_dataset,
        steps_persamples_for_each_song_song=np.array(samples_for_each_song),
        steps_per_second=np.array([steps_per_second]),
        freq_buckets=freq_buckets,
        allow_pickle=False,
    )

def clean_and_store_data(dest_file_path, raw_data_dir):
    paths: list[str] = mp3s_in_dir(raw_data_dir)
    dbcqts, steps_per_d = dbcqts_from_paths(paths=paths)
    samples_for_each_song = steps_per_cqt(dbcqts)
    dataset = standardized_dataset(dbcqts=dbcqts)
    print(f"============ Reshaping after standardization to {DURATION} second intervals ============")
    print("Steps per duration:", steps_per_d)
    new_dataset = dataset.reshape((int(dataset.shape[0] / steps_per_d), int(steps_per_d * dataset.shape[1])))
    print("New dataset shape", new_dataset.shape)
    print(f"============ Finished reshaping ============")
    save_data(
        dest_file=dest_file_path,
        core_dataset=new_dataset,
        samples_for_each_song=samples_for_each_song,
        steps_per_duration=steps_per_d,
        freq_buckets=FREQ,
    )

def retrieve_data(source_file_path: str):
    result = np.load(file=source_file_path)
    print(len(result))