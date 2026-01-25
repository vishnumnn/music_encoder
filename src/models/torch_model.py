import torch
DEVICE = torch.device("cuda")


class TorchModel(torch.nn.Module):
    def __init__(self, in_features, latent_dim):
        super(TorchModel, self).__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_features=in_features, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, out_features=in_features)

    def forward(self, input):
        mu, log_var = self.encoder(input)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(DEVICE)
        return mu + eps * std

    def x_to_z(self, x):
        with torch.no_grad():
          mu, log_var = self.encoder(x)
          z = self.reparameterize(mu, log_var)
          return z

    def details(self):
        return f"vae_in_{self.in_features}_out_{self.latent_dim}"

class Encoder(torch.nn.Module):
    def __init__(self, in_features, latent_dim):
        super(Encoder, self).__init__()
        self.fc_1 = torch.nn.Linear(in_features=in_features, out_features=256)
        self.fc_2 = torch.nn.Linear(in_features=256, out_features=128)
        self.fc_mu = torch.nn.Linear(in_features=128, out_features=latent_dim)
        self.fc_log_var = torch.nn.Linear(in_features=128, out_features=latent_dim)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.fc_1(x))
        x = self.leaky_relu(self.fc_2(x))
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, out_features):
        super(Decoder, self).__init__()
        self.fc_1 = torch.nn.Linear(in_features=latent_dim, out_features=128)
        self.fc_2 = torch.nn.Linear(in_features=128, out_features=out_features)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z):
        h = self.leaky_relu(self.fc_1(z))
        x_hat = self.sigmoid(self.fc_2(h))
        return x_hat

def loss_function(x_hat, x, mu, log_var):
    reconstruction_loss = torch.nn.functional.mse_loss(x_hat, x)
    regularization_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + regularization_loss