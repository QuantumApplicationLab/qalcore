import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
from .data import balance_dataset, extract_features, get_normalization_data, normalize
import matplotlib.pyplot as plt


@qml.qnode(qml.device("default.qubit", wires=1))
def reupload_circuit(params, x, y):
    """A variational quantum circuit representing the Universal classifier.

    Args:
        params (array[float]): array of parameters
        x (array[float]): single input vector
        y (array[float]): single output state density matrix

    Returns:
        float: fidelity between output state and input
    """
    
    n_inp_bloc = -(len(x)//-3)
    
    for p in params:
        for i in range(n_inp_bloc):
            qml.Rot(*x[i*3:(i+1)*3], wires=0)
        qml.Rot(*p, wires=0)
    return qml.expval(qml.Hermitian(y, wires=[0]))

class ReuploadClassifier:

    def __init__(self, circuit, train_dataset, valid_dataset, num_layers ):
        self.qcircuit = circuit
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.num_layers = num_layers
        self.state_labels = np.array([[[1], [0]], [[0], [1]]], requires_grad=False)


    def process_data(self, npts = 'all', features='all', normalization=True):

        if isinstance(npts, int):
            self.train_dataset = balance_dataset(self.train_dataset, npts)
            self.valid_dataset = balance_dataset(self.valid_dataset, npts)

        # only consider a subset of the features
        if isinstance(features, list):
            self.train_dataset = extract_features(self.train_dataset, features)
            self.valid_dataset = extract_features(self.valid_dataset, features)

        # normalize the data between 0 and 2pi
        if normalization:
            normalization_data = get_normalization_data(self.train_dataset)
            self.train_dataset = normalize(self.train_dataset, normalization_data)
            self.valid_dataset = normalize(self.valid_dataset, normalization_data)

        # make int labels 
        self.train_dataset.labels = np.array(self.train_dataset.labels.astype(int), requires_grad=False)
        self.train_dataset.features = np.array(self.train_dataset.features, requires_grad=False)
        self.valid_dataset.labels = np.array(self.valid_dataset.labels.astype(int), requires_grad=False)
        self.valid_dataset.features = np.array(self.valid_dataset.features, requires_grad=False)


    # Define output labels as quantum state vectors
    @staticmethod
    def density_matrix(state):
        """Calculates the density matrix representation of a state.

        Args:
            state (array[complex]): array representing a quantum state vector

        Returns:
            dm: (array[complex]): array representing the density matrix
        """
        return state * np.conj(state).T
  
    def cost(self, params, x, y, state_labels=None):
        """Cost function to be minimized.

        Args:
            params (array[float]): array of parameters
            x (array[float]): 2-d array of input vectors
            y (array[float]): 1-d array of targets
            state_labels (array[float]): array of state representations for labels

        Returns:
            float: loss value to be minimized
        """
        # Compute prediction for each input in data batch
        loss = 0.0
        dm_labels = [self.density_matrix(s) for s in state_labels]
        for i in range(len(x)):
            f = self.qcircuit(params, x[i], dm_labels[y[i]])
            loss = loss + (1 - f) ** 2
        return loss / len(x)
    
    def test(self, params, x, y, state_labels=None):
        """
        Tests on a given set of data.

        Args:
            params (array[float]): array of parameters
            x (array[float]): 2-d array of input vectors
            y (array[float]): 1-d array of targets
            state_labels (array[float]): 1-d array of state representations for labels

        Returns:
            predicted (array([int]): predicted labels for test data
            output_states (array[float]): output quantum states from the circuit
        """
        fidelity_values = []
        dm_labels = [self.density_matrix(s) for s in state_labels]
        predicted = []

        for i in range(len(x)):
            fidel_function = lambda y: self.qcircuit(params, x[i], y)
            fidelities = [fidel_function(dm) for dm in dm_labels]
            best_fidel = np.argmax(fidelities)

            predicted.append(best_fidel)
            fidelity_values.append(fidelities)

        return np.array(predicted), np.array(fidelity_values)
    
    @staticmethod
    def accuracy_score(y_true, y_pred):
        """Accuracy score.

        Args:
            y_true (array[float]): 1-d array of targets
            y_predicted (array[float]): 1-d array of predictions
            state_labels (array[float]): 1-d array of state representations for labels

        Returns:
            score (float): the fraction of correctly classified samples
        """
        score = y_true == y_pred
        return score.sum() / len(y_true)
    
    @staticmethod
    def iterate_minibatches(inputs, targets, batch_size):
        """
        A generator for batches of the input data

        Args:
            inputs (array[float]): input data
            targets (array[float]): targets

        Returns:
            inputs (array[float]): one batch of input data of length `batch_size`
            targets (array[float]): one batch of targets of length `batch_size`
        """
        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield inputs[idxs], targets[idxs]

    @staticmethod
    def pad_data(data):
        if data.shape[1]%3 != 0:
            nadd = 3-data.shape[1]%3
            data = np.hstack((data, np.zeros((data.shape[0], nadd), requires_grad=False)))
        return data
    

    def train(self, epochs = 25, batch_size = 32, opt =  AdamOptimizer(0.6, beta1=0.9, beta2=0.999)):

        # Generate training and test data)
        X_train = self.pad_data(self.train_dataset.features)
        y_train = self.train_dataset.labels


        X_test = self.pad_data(self.valid_dataset.features)
        y_test = self.valid_dataset.labels
       
        # initialize random weights
        params = np.random.uniform(size=(self.num_layers, 3), requires_grad=True)

        predicted_train, fidel_train = self.test(params, X_train, y_train, self.state_labels)
        accuracy_train = self.accuracy_score(y_train, predicted_train)

        predicted_test, fidel_test = self.test(params, X_test, y_test, self.state_labels)
        accuracy_test = self.accuracy_score(y_test, predicted_test)

        # save predictions with random weights for comparison
        initial_predictions = predicted_test
        loss = self.cost(params, X_test, y_test, self.state_labels)

        print(
            "Epoch: {:2d} | Cost: {:3f} | Train accuracy: {:3f} | Test Accuracy: {:3f}".format(
                0, loss, accuracy_train, accuracy_test
            )
        )

        for it in range(epochs):
            for Xbatch, ybatch in self.iterate_minibatches(X_train, y_train, batch_size=batch_size):
                params, _, _, _ = opt.step(self.cost, params, Xbatch, ybatch, self.state_labels)

            predicted_train, fidel_train = self.test(params, X_train, y_train, self.state_labels)
            accuracy_train = self.accuracy_score(y_train, predicted_train)
            loss = self.cost(params, X_train, y_train, self.state_labels)

            predicted_test, fidel_test = self.test(params, X_test, y_test, self.state_labels)
            accuracy_test = self.accuracy_score(y_test, predicted_test)
            res = [it + 1, loss, accuracy_train, accuracy_test]
            print(
                "Epoch: {:2d} | Loss: {:3f} | Train accuracy: {:3f} | Test accuracy: {:3f}".format(
                    *res
                )
            )


        print("Learned weights")
        for i in range(self.num_layers):
            print("Layer {}: {}".format(i, params[i]))

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        self.plot_data(X_test, initial_predictions, fig, axes[0])
        self.plot_data(X_test, predicted_test, fig, axes[1])
        self.plot_data(X_test, y_test, fig, axes[2])
        axes[0].set_title("Predictions with random weights")
        axes[1].set_title("Predictions after training")
        axes[2].set_title("True test data")
        plt.tight_layout()
        plt.show()

        return params
    
    @staticmethod
    def plot_data(x, y, fig=None, ax=None):
        """
        Plot data with red/blue values for a binary classification.

        Args:
            x (array[tuple]): array of data points as tuples
            y (array[int]): array of data points as tuples
        """
        if fig == None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        reds = y == 0
        blues = y == 1
        ax.scatter(x[reds, 0], x[reds, 1], c="red", s=20, edgecolor="k", alpha=0.1)
        ax.scatter(x[blues, 0], x[blues, 1], c="blue", s=20, edgecolor="k", alpha=0.1)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")