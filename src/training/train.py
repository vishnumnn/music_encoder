from src.models.torch_model import TorchModel
from src.training.train_dataset import TrainDataset
import torch


BASE_EPOCHS = 100


class Trainer:
    def __init__(
        self,
        epochs: int,
        dataset: TrainDataset,
        model: TorchModel,
        checkpoint_dir: str,
    ) -> None:
        self.epochs = epochs
        self.dataset = dataset
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def build_from_checkpoint(self, in_features: int, latent_dim: int, path_to_checkpoint: str, dataset: TrainDataset):
        checkpoint = torch.load(path_to_checkpoint, weights_only=True)
        model: TorchModel = TorchModel(in_features=in_features, latent_dim=latent_dim).load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters()).load_state_dict(checkpoint['optimizer_state_dict'])
        epochs = max(1, BASE_EPOCHS - checkpoint['epoch'])
        print(f"Only {epochs} left for training")
        return Trainer(
            epochs=epochs,
            dataset=dataset,
            model=model,
            optimizer=optimizer,
        )

    def loss(self, x_hat, x, mu, log_var):
        reconstruction_loss = torch.nn.functional.mse_loss(x_hat, x)
        regularization_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + regularization_loss

    def checkpoint(self, epoch: int, loss):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            },
            f"{self.checkpoint_dir}/{self.model.details()}_{epoch}.pth"
        )

    def train(self):
        for i in range(self.epochs):
            total_loss = 0
            for j, (x, _) in enumerate(self.dataset):
                self.optimizer.zero_grad()
                x_hat, mu, log_var = self.model(x)
                loss = self.loss(x_hat, x, mu, log_var)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

                encoder = self.model.encoder
                print(encoder.fc_1.weight)
                print(encoder.fc_2.weight)
                print(encoder.fc_mu.weight)
                print(encoder.fc_log_var.weight)
                decoder = self.model.decoder
                print(decoder.fc_1.weight)
                print(decoder.fc_2.weight)
                exit()
            if (i + 1) % 10 == 0:
                print(f"===== Checkpointing epoch {i} =====")
                print(f"Loss: {loss.item()}, model_params: {self.model.details()}")
                self.checkpoint(i, loss)
            print(f"Epoch {i+1}/{self.epochs}, Average Loss: {total_loss/len(self.dataset)}, Total Loss: {total_loss}")