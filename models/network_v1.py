import math
import numpy as np
from gym_derk import ObservationKeys


class Network:
    def __init__(self, weights=None, biases=None):
        self.network_outputs = 13
        if weights is None:
            weights_shape = (self.network_outputs, len(ObservationKeys))
            self.weights = np.random.normal(size=weights_shape)
        else:
            self.weights = weights
        if biases is None:
            self.biases = np.random.normal(size=(self.network_outputs))
        else:
            self.biases = biases

    def clone(self):
        return Network(np.copy(self.weights), np.copy(self.biases))

    def forward(self, observations):
        outputs = np.add(np.matmul(self.weights, observations), self.biases)
        casts = outputs[3:6]
        cast_i = np.argmax(casts)
        focuses = outputs[6:13]
        focus_i = np.argmax(focuses)
        return (
            math.tanh(outputs[0]), # MoveX
            math.tanh(outputs[1]), # Rotate
            max(min(outputs[2], 1), 0), # ChaseFocus
            (cast_i + 1) if casts[cast_i] > 0 else 0, # CastSlot
            (focus_i + 1) if focuses[focus_i] > 0 else 0, # Focus
        )

    def copy_and_mutate(self, network, mr=0.1):
        self.weights = np.add(network.weights, np.random.normal(size=self.weights.shape) * mr)
        self.biases = np.add(network.biases, np.random.normal(size=self.biases.shape) * mr)
