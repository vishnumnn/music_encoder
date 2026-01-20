from argparse import ArgumentParser
from src.models.archive import Archive
from src.preprocessing.clean_data import clean_and_store_data, retreive_data


DIR = '/Users/vmenon/repos/music_encoder/data/raw'
DEST_FILE_PATH = '/Users/vmenon/repos/music_encoder/data/clean/compressed_combined_data_set'


if __name__ == '__main__':
    parser = ArgumentParser(description="A script to read data mp3 data and write cleaned and compressed output")
    parser.add_argument("--dest_file_path", default=DEST_FILE_PATH)
    parser.add_argument("--raw_data_dir", default=DIR)
    args = parser.parse_args()
    clean_and_store_data(dest_file_path=args.dest_file_path, raw_data_dir=args.raw_data_dir)
    archive: Archive = retreive_data(f"{DEST_FILE_PATH}.npz")