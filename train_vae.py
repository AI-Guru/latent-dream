import numpy as np
import pickle
import vae_model
import matplotlib.pyplot as plt

# Load data.
dataset_path = "vae_dataset-10000.p"
print("Loading dataset from {}...".format(dataset_path))
observations = pickle.load(open(dataset_path, "rb"))

# Hyperparameters for ConvVAE
z_size=64
batch_size=100
learning_rate=0.0001
kl_tolerance=0.5

# Parameters for training
NUM_EPOCHS = 100
DATA_DIR = "record"

# TODO can this move?
vae_model.reset_graph()

vae = vae_model.VariationalAutoencoder(
    z_size=z_size,
    batch_size=batch_size,
    learning_rate=learning_rate,
    kl_tolerance=kl_tolerance,
    is_training=True,
    reuse=False,
    gpu_mode=True)

# Train the VAE.
#observations = observations[0:400]
history = vae.train(dataset=observations, num_epochs=NUM_EPOCHS)

# Render losses.
for key in ["episode_train_loss", "episode_r_loss", "episode_kl_loss"]:
    plt.plot(history[key], label=key)
plt.legend()
plot_path = "vae-episode-losses.png"
plt.savefig(plot_path)
plt.close()
print("Written plot to {}.".format(plot_path))

# Render losses.
for key in ["batch_train_loss", "batch_r_loss", "batch_kl_loss"]:
    plt.plot(history[key], label=key)
plt.legend()
plot_path = "vae-batch-losses.png"
plt.savefig(plot_path)
plt.close()
print("Written plot to {}.".format(plot_path))

# Save model.
model_path = "vae.json"
vae.save_model("models")
print("Model saved to {}.".format(model_path))
