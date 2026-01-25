from argparse import ArgumentParser

from src.models.archive import Archive
from src.models.torch_model import TorchModel
from src.preprocessing.clean_data import INPUT_DIM, retrieve_data
from src.training.train import BASE_EPOCHS, Trainer
from src.training.train_dataset import TrainDataset


LATENT_DIM=15


if __name__ == '__main__':
    parser = ArgumentParser(description="Load and train on training data at dir")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--use_checkpoint")
    parser.add_argument("--train_data_path")
    args = parser.parse_args()
    archive: Archive = retrieve_data(args.train_data_path)
    train_dataset: TrainDataset = TrainDataset.from_archive(
        archive=archive
    )
    model: TorchModel = TorchModel(
        in_features=INPUT_DIM,
        latent_dim=LATENT_DIM,
    )
    trainer: Trainer
    if parser.use_checkpoint:
        trainer = Trainer.build_from_checkpoint(
            in_features=INPUT_DIM,
            latent_dim=LATENT_DIM,
            path_to_checkpoint=parser.use_checkpoint,
            dataset=train_dataset,
        )
    else:
        trainer = Trainer(
            epochs=BASE_EPOCHS,
            dataset=train_dataset,
            model=model,
            checkpoint_dir=parser.checkpoint_dir,
        )
    print("begin training model")
    trainer.train()
