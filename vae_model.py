import tensorflow as tf
import json
import numpy as np
from tqdm import tqdm
import os

class VariationalAutoencoder(object):

    def __init__(self, z_size=64, batch_size=100, learning_rate=0.0001, kl_tolerance=0.5, is_training=True, reuse=False, gpu_mode=True, use_logistic_loss=False):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.kl_tolerance = kl_tolerance
        self.reuse = reuse
        self.use_logistic_loss = use_logistic_loss
        with tf.variable_scope('conv_vae', reuse=self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu.')
                    self.build_graph()
            else:
                tf.logging.info('Model using gpu.')
                self.build_graph()
        self.initialize_session()


    def build_graph(self):
        """
        Builds the Tensorflow graph.
        """

        self.graph = tf.Graph()
        with self.graph.as_default():

            # Input.
            self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="input")

            # Create the encoder.
            h = tf.layers.conv2d(self.x, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
            h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
            h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
            h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
            h = tf.reshape(h, [-1, 2*2*256])

            # Create the variational part.
            self.mu = tf.layers.dense(h, self.z_size, name="enc_fc_mu")
            self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_log_var")
            self.sigma = tf.exp(self.logvar / 2.0)
            self.epsilon = tf.random_normal([self.batch_size, self.z_size])
            self.z = self.mu + self.sigma * self.epsilon

            # Create the decoder.
            h = tf.layers.dense(self.z, 4 * 256, name="dec_fc")
            h = tf.reshape(h, [-1, 1, 1, 4 * 256])
            h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
            h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
            h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")
            self.y = tf.layers.conv2d_transpose(h, 3, 6, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")

            # Training operations.
            if self.is_training:

                # Global step.
                self.global_step = tf.Variable(0, name="global_step", trainable=False)

                # Reconstruction Loss.
                if self.use_logistic_loss == True:
                    epsilon = 1e-6
                    self.r_loss = - tf.reduce_mean(
                        self.x * tf.log(self.y + epsilon) + (1.0 - self.x) * tf.log(1.0 - self.y + epsilon),
                        reduction_indices = [1,2,3]
                    )
                    self.r_loss = tf.reduce_mean(self.r_loss) * 64.0 * 64.0
                else:
                    self.r_loss = tf.reduce_sum(
                        tf.square(self.x - self.y),
                        reduction_indices = [1,2,3]
                    )
                    self.r_loss = tf.reduce_mean(self.r_loss)

                # Augmented kl-loss per dimension.
                self.kl_loss = - 0.5 * tf.reduce_sum(
                    (1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)),
                    reduction_indices = 1
                )
                self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
                self.kl_loss = tf.reduce_mean(self.kl_loss)

                # Overall loss.
                self.loss = self.r_loss + self.kl_loss

                # Set learning rate and optimizer.
                self.lr = tf.Variable(self.learning_rate, trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                grads = self.optimizer.compute_gradients(self.loss) # can potentially clip gradients here.

                # Set training operation.
                self.train_operation = self.optimizer.apply_gradients(
                    grads,
                    global_step=self.global_step,
                    name="train_step")

            # Initialize variables.
            self.init = tf.global_variables_initializer()

            # Create assign opsfor VAE.
            t_vars = tf.trainable_variables()
            self.assign_operations = {}
            for var in t_vars:
                    if var.name.startswith("conv_vae"):
                            pshape = var.get_shape()
                            pl = tf.placeholder(tf.float32, pshape, var.name[:-2] + "_placeholder")
                            assign_op = var.assign(pl)
                            self.assign_operations[var] = (assign_op, pl)


    def initialize_session(self):
        """
        Initializes the Tensorflow session.
        """

        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)


    def close_session(self):
        """
        Closes the TensorFlow session.
        """

        self.session.close()


    def encode(self, x):
        """
        Encodes an input into latent space as z.
        """

        return self.session.run(self.z, feed_dict={self.x: x})


    def encode_mu_logvar(self, x):
        """
        Encodes an input into latent space as mu and logvar.
        """

        (mu, logvar) = self.session.run([self.mu, self.logvar], feed_dict={self.x: x})
        return mu, logvar


    def decode(self, z):
        """
        Decodes a latent space vector.
        """

        return self.session.run(self.y, feed_dict={self.z: z})


    def reconstruct(self, x):
        """
        Reconstructs an image.
        """

        return self.session.run(self.y, feed_dict={self.x: x})


    def get_trainable_parameters(self):
        """
        Gets the trainable parameters.
        """

        model_names = []
        model_parameters = []
        model_shapes = []
        with self.graph.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                if var.name.startswith('conv_vae'):
                    parameter_name = var.name
                    p = self.session.run(var)
                    model_names.append(parameter_name)
                    parameters = np.round(p*10000).astype(np.int).tolist()
                    model_parameters.append(parameters)
                    model_shapes.append(p.shape)
        return model_parameters, model_shapes, model_names


    def get_random_trainable_parameters(self, stdev=0.5):
        """
        Gets random trainable parameters.
        """

        _, model_shapes, _ = self.get_trainable_parameters()
        random_parameters = []
        for model_shape in model_shapes:
            random_parameters.append(np.random.standard_cauchy(model_shape) * stdev) # spice things up!
        return random_parameters


    def set_model_parameters(self, parameters):
        """
        Sets the model parameters.
        """

        with self.graph.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                if var.name.startswith('conv_vae'):
                    pshape = tuple(var.get_shape().as_list())
                    p = np.array(parameters[idx])
                    assert pshape == p.shape, "inconsistent shape"
                    assign_op, pl = self.assign_operations[var]
                    self.session.run(assign_op, feed_dict={pl.name: p / 10000.0})
                    idx += 1

    def load_json(self, jsonfile='vae.json'):
        """
        Loads the model parameters from JSON.
        """

        with open(jsonfile, 'r') as f:
            parameters = json.load(f)
        self.set_model_parameters(parameters)

    def save_json(self, jsonfile='vae.json'):
        """
        Saves the model to JSON.
        """

        model_parameters, model_shapes, model_names = self.get_trainable_parameters()
        qparameters = []
        for p in model_parameters:
            qparameters.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparameters, outfile, sort_keys=True, indent=0, separators=(',', ': '))


    def set_random_parameters(self, stdev=0.5):
        """
        Sets the model parameters to random ones.
        """

        random_parameters = self.get_random_trainable_parameters(stdev)
        self.set_model_parameters(random_parameters)


    def save_model(self, model_save_path):
        """
        Saves the model.
        """

        sess = self.session
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(model_save_path, 'vae')
        tf.logging.info('saving model %s.', checkpoint_path)
        saver.save(sess, checkpoint_path, 0) # just keep one

    def load_checkpoint(self, checkpoint_path):
        """
        Loads the model from a checkpoint.
        """

        session = self.session
        with self.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        print("Loading model", checkpoint.model_checkpoint_path)
        tf.logging.info("Loading model {}.".format(checkpoint.model_checkpoint_path))
        saver.restore(session, checkpoint.model_checkpoint_path)

    def train(self, dataset, num_epochs):
        """
        Trains the model on a dataset.
        """

        total_length = len(dataset)
        num_batches = int(np.floor(total_length/self.batch_size))

        # Create history.
        history = {}
        history["batch_train_loss"] = []
        history["batch_r_loss"] = []
        history["batch_kl_loss"] = []
        history["episode_train_loss"] = []
        history["episode_r_loss"] = []
        history["episode_kl_loss"] = []

        # Train.
        print("Training on {} samples...".format(len(dataset)))
        for epoch in range(num_epochs):
            np.random.shuffle(dataset)
            episode_train_loss = 0.0
            episode_r_loss = 0.0
            episode_kl_loss = 0.0
            for idx in range(num_batches):
                batch = dataset[idx*self.batch_size:(idx+1)*self.batch_size]

                feed = {self.x: batch,}

                # Gradient descent.
                (batch_train_loss, batch_r_loss, batch_kl_loss, _, _) = self.session.run([
                    self.loss, self.r_loss, self.kl_loss, self.global_step, self.train_operation
                ], feed)

                # Store in history.
                history["batch_train_loss"].append(batch_train_loss)
                history["batch_r_loss"].append(batch_r_loss)
                history["batch_kl_loss"].append(batch_kl_loss)

                # Collect losses.
                episode_train_loss += batch_train_loss
                episode_r_loss += batch_r_loss
                episode_kl_loss += batch_kl_loss

                # Print mid episode.
                print("Episode:{}/{} Step:{}/{} Loss:{} R-Loss:{} KL-Loss:{}\t".format(epoch + 1, num_epochs, idx + 1, num_batches, batch_train_loss, batch_r_loss, batch_kl_loss), end="\r")

            # Average losses.
            episode_train_loss /= num_batches
            episode_r_loss /= num_batches
            episode_kl_loss /= num_batches

            # Print at end of episode.
            print("Episode:{}/{} Step:{}/{} Loss:{} R-Loss:{} KL-Loss:{}\t".format(epoch + 1, num_epochs, idx + 1, num_batches, episode_train_loss, episode_r_loss, episode_kl_loss))

            # Store in history.
            history["episode_train_loss"].append(episode_train_loss)
            history["episode_r_loss"].append(episode_r_loss)
            history["episode_kl_loss"].append(episode_kl_loss)

        # Done.
        return history


def reset_graph():
    if "sess" in globals() and sess:
        sess.close()
    tf.reset_default_graph()
