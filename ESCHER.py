# Original code from OpenSpiel Deep CFR implementation:
# https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/deep_cfr_tf2.py
# Original Deep CFR code copyright 2019 DeepMind Technologies Limited
# ESCHER Code copyright 2022 Stephen McAleer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements ESCHER.


The algorithm defines `regret`, `value`, and `average policy` networks.
The regret network is trained to estimate the cumulative regret for an infostate-action pair.
The policy at each timestep comes from performing regret matching on the estimated cumulative regret.
The value network is trained to estimate the value of a game under the current joint policy conditioned
on a history (state).
The average policy network is trained to estimate the average policy over all timesteps.
To train these networks we use three reservoir buffers, one for each network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import collections
import contextlib
import os
import random
import numpy as np
import tensorflow as tf

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

import pyspiel
import time

# The size of the shuffle buffer used to reshuffle part of the data each
# epoch within one training iteration
REGRET_TRAIN_SHUFFLE_SIZE = 100000
VALUE_TRAIN_SHUFFLE_SIZE = 100000
AVERAGE_POLICY_TRAIN_SHUFFLE_SIZE = 1000000


def battleship_history_tensor(state):
    """
    Important! Only works for
    board_width = 2,
    board_height = 2,
    ship_sizes = [1;2],
    num_shots = 4,
    allow_repeated_shots = False
    """

    # first 4 spots are location of a for player 0
    # next 4 spots are location of b for player 0
    # next 4 spots are all of player 0's actions
    # then repeat for player 1
    history = state.history()
    history_tensor = np.zeros(2 * (4 + 4 + 4))
    b_map = {4: 0, 6: 1, 8: 2, 9: 3}

    for i in range(len(history)):
        if i == 0:  # a for player 0
            action = history[i] - 4
            history_tensor[action] = 1
        elif i == 1:  # a for player 1
            action = history[i] - 4
            history_tensor[12 + action] = 1
        elif i == 2:  # b for player 0
            action = b_map[history[i]]
            history_tensor[4 + action] = 1
        elif i == 3:
            action = b_map[history[i]]
            history_tensor[12 + 4 + action] = 1
        else:
            action = history[i]
            if i % 2 == 0:
                history_tensor[8 + action] = 1
            else:
                history_tensor[12 + 8 + action] = 1
    return history_tensor


def battleship_infostate_tensor(state, player):
    """
    Important! Only works for
    board_width = 2,
    board_height = 2,
    ship_sizes = [1;2],
    num_shots = 4,
    allow_repeated_shots = False
    """
    # first 4 spots are location of a for player
    # next 4 spots are location of b for player
    # next 4 spots are for player's actions that are hits
    # next 4 spots are for player's actions that are water
    # next 4 spots are for player's actions that are sinks
    # next 4 spots are all of other player's actions

    string_to_index_map = {
        '0_0': 0,
        '0_1': 1,
        '1_0': 2,
        '1_1': 3,
    }

    info_tensor = np.zeros(6 * 4)
    info_list = state.information_state_string(player).split("/")

    for i in range(len(info_list)):
        info = info_list[i]

        if i == 1:
            assert info[0] == 'h' or info[0] == 'v'
            index = string_to_index_map[info[-3:]]
            info_tensor[index] = 1
        if i == 2:
            assert info[0] == 'h' or info[0] == 'v'
            index = string_to_index_map[info[-3:]]
            info_tensor[index + 4] = 1
        if i > 2:
            if info[0] == 's':
                index = string_to_index_map[info[-5:-2]]
                if info[-1] == 'H':
                    info_tensor[index + 2 * 4] = 1
                elif info[-1] == 'W':
                    info_tensor[index + 3 * 4] = 1
                elif info[-1] == 'S':
                    info_tensor[index + 4 * 4] = 1
                else:
                    print(info, "unexpected infostate")
                    return None
            elif info[0] == 'o':
                index = string_to_index_map[info[-3:]]
                info_tensor[index + 5 * 4] = 1
            else:
                print(info, 'unexpected infostate')
                return None
    return info_tensor


def get_oshi_hist_obs(obs_tensor, current_player, num_coins, last_action):
    last_action_array = np.zeros(num_coins + 1)
    if current_player == 1:
        assert obs_tensor[1] == 1
        last_action_array[last_action] = 1
    return np.append(obs_tensor, last_action_array)


def get_markov_soccer_hist_obs(obs_tensor, current_player, last_action):
    last_action_array = np.zeros(5)
    if current_player == 1:
        assert obs_tensor[1] == 1
        last_action_array[last_action] = 1
    return np.append(obs_tensor, last_action_array)


class ReservoirBuffer(object):
    """Allows uniform sampling over a stream of data.

    This class supports the storage of arbitrary elements, such as observation
    tensors, integer actions, etc.

    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    """

    def __init__(self, reservoir_buffer_capacity):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        """Potentially adds `element` to the reservoir buffer.

        Args:
          element: data to be added to the reservoir buffer.
        """
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        """Returns `num_samples` uniformly sampled from the buffer.

        Args:
          num_samples: `int`, number of samples to draw.

        Returns:
          An iterable over `num_samples` random elements of the buffer.

        Raises:
          ValueError: If there are less than `num_samples` elements in the buffer
        """
        if len(self._data) < num_samples:
            raise ValueError('{} elements could not be sampled from size {}'.format(
                num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def clear(self):
        self._data = []
        self._add_calls = 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    @property
    def data(self):
        return self._data

    def get_data(self):
        return self._data

    def shuffle_data(self):
        random.shuffle(self._data)

    def get_num_calls(self):
        return self._add_calls


class SkipDense(tf.keras.layers.Layer):
    """Dense Layer with skip connection."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.hidden = tf.keras.layers.Dense(units, kernel_initializer='he_normal')

    def call(self, x):
        return self.hidden(x) + x


class PolicyNetwork(tf.keras.Model):
    """Implements the policy network as an MLP.

    Implements the policy network as a MLP with skip connections in adjacent
    layers with the same number of units, except for the last hidden connection
    where a layer normalization is applied.
    """

    def __init__(self,
                 input_size,
                 policy_network_layers,
                 num_actions,
                 activation='leakyrelu',
                 **kwargs):
        super().__init__(**kwargs)
        self._input_size = input_size
        self._num_actions = num_actions
        if activation == 'leakyrelu':
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        elif activation == 'relu':
            self.activation = tf.keras.layers.ReLU()
        else:
            self.activation = activation

        self.softmax = tf.keras.layers.Softmax()

        self.hidden = []
        prevunits = 0
        for units in policy_network_layers[:-1]:
            if prevunits == units:
                self.hidden.append(SkipDense(units))
            else:
                self.hidden.append(
                    tf.keras.layers.Dense(units, kernel_initializer='he_normal'))
            prevunits = units
        self.normalization = tf.keras.layers.LayerNormalization()
        self.lastlayer = tf.keras.layers.Dense(
            policy_network_layers[-1], kernel_initializer='he_normal')

        self.out_layer = tf.keras.layers.Dense(num_actions)

    @tf.function
    def call(self, inputs):
        """Applies Policy Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Action probabilities
        """
        x, mask = inputs
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)

        x = self.normalization(x)
        x = self.lastlayer(x)
        x = self.activation(x)
        x = self.out_layer(x)
        x = tf.where(mask == 1, x, -10e20)
        x = self.softmax(x)
        return x


class RegretNetwork(tf.keras.Model):
    """Implements the regret network as an MLP.

    Implements the regret network as an MLP with skip connections in
    adjacent layers with the same number of units, except for the last hidden
    connection where a layer normalization is applied.
    """

    def __init__(self,
                 input_size,
                 regret_network_layers,
                 num_actions,
                 activation='leakyrelu',
                 **kwargs):
        super().__init__(**kwargs)
        self._input_size = input_size
        self._num_actions = num_actions
        if activation == 'leakyrelu':
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        elif activation == 'relu':
            self.activation = tf.keras.layers.ReLU()
        else:
            self.activation = activation

        self.hidden = []
        prevunits = 0
        for units in regret_network_layers[:-1]:
            if prevunits == units:
                self.hidden.append(SkipDense(units))
            else:
                self.hidden.append(
                    tf.keras.layers.Dense(units, kernel_initializer='he_normal'))
            prevunits = units
        self.normalization = tf.keras.layers.LayerNormalization()
        self.lastlayer = tf.keras.layers.Dense(
            regret_network_layers[-1], kernel_initializer='he_normal')

        self.out_layer = tf.keras.layers.Dense(num_actions)

    @tf.function
    def call(self, inputs):
        """Applies Regret Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Cumulative regret for each info_state action
        """
        x, mask = inputs
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)

        x = self.normalization(x)
        x = self.lastlayer(x)
        x = self.activation(x)
        x = self.out_layer(x)
        x = mask * x

        return x


class ValueNetwork(tf.keras.Model):
    """Implements the history value network as an MLP.

    Implements the history value network as an MLP with skip connections in
    adjacent layers with the same number of units, except for the last hidden
    connection where a layer normalization is applied.
    """

    def __init__(self,
                 input_size,
                 val_network_layers,
                 activation='leakyrelu',
                 **kwargs):
        super().__init__(**kwargs)
        self._input_size = input_size
        if activation == 'leakyrelu':
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        elif activation == 'relu':
            self.activation = tf.keras.layers.ReLU()
        else:
            self.activation = activation

        self.hidden = []
        prevunits = 0
        for units in val_network_layers[:-1]:
            if prevunits == units:
                self.hidden.append(SkipDense(units))
            else:
                self.hidden.append(
                    tf.keras.layers.Dense(units, kernel_initializer='he_normal'))
            prevunits = units
        self.normalization = tf.keras.layers.LayerNormalization()
        self.lastlayer = tf.keras.layers.Dense(
            val_network_layers[-1], kernel_initializer='he_normal')

        self.out_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs):
        """Applies Value Network.

        Args:
            inputs: Tuple representing (info_state, legal_action_mask)

        Returns:
            Cumulative regret for each info_state action
        """
        x, mask = inputs
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)

        x = self.normalization(x)
        x = self.lastlayer(x)
        x = self.activation(x)
        x = self.out_layer(x)

        return x


class ESCHERSolver(policy.Policy):
    def __init__(self,
                 game,
                 policy_network_layers=(256, 128),
                 regret_network_layers=(256, 128),
                 value_network_layers=(256, 128),
                 num_iterations: int = 100,
                 num_traversals: int = 130000,
                 num_val_fn_traversals: int = 100,
                 learning_rate: float = 1e-3,
                 batch_size_regret: int = 10000,
                 batch_size_value: int = 2024,
                 batch_size_average_policy: int = 10000,
                 markov_soccer: bool = False,
                 phantom_ttt=False,
                 dark_hex: bool = False,
                 memory_capacity: int = int(1e5),
                 policy_network_train_steps: int = 15000,
                 regret_network_train_steps: int = 5000,
                 value_network_train_steps: int = 4048,
                 check_exploitability_every: int = 20,
                 reinitialize_regret_networks: bool = True,
                 reinitialize_value_network: bool = True,
                 save_regret_networks: str = None,
                 append_legal_actions_mask=False,
                 save_average_policy_memories: str = None,
                 save_policy_weights=True,
                 expl: float = 1.0,
                 val_expl: float = 0.01,
                 importance_sampling_threshold: float = 100.,
                 importance_sampling: bool = True,
                 clear_value_buffer: bool = True,
                 val_bootstrap=False,
                 oshi_zumo=False,
                 use_balanced_probs: bool = False,
                 battleship=False,
                 starting_coins=8,
                 val_op_prob=0.,
                 infer_device='cpu',
                 debug_val=False,
                 play_against_random=False,
                 train_device='cpu',
                 experiment_string=None,
                 all_actions=True,
                 random_policy_path=None,
                 *args, **kwargs):
        """Initialize the ESCHER algorithm.

        Args:
          game: Open Spiel game.
          policy_network_layers: (list[int]) Layer sizes of average_policy net MLP.
          regret_network_layers: (list[int]) Layer sizes of regret net MLP.
          value_network_layers: (list[int]) Layer sizes of value net MLP.
          num_iterations: Number of iterations.
          num_traversals: Number of traversals per iteration.
          num_val_fn_traversals: Number of history value function traversals per iteration
          learning_rate: Learning rate.
          batch_size_regret: (int) Batch size to sample from regret memories.
          batch_size_average_policy: (int) Batch size to sample from average_policy memories.
          memory_capacity: Number of samples that can be stored in memory.
          policy_network_train_steps: Number of policy network training steps (one
            policy training iteration at the end).
          regret_network_train_steps: Number of regret network training steps
            (per iteration).
          reinitialize_regret_networks: Whether to re-initialize the regret
            network before training on each iteration.
          save_regret_networks: If provided, all regret network itearations
            are saved in the given folder. This can be useful to implement SD-CFR
            https://arxiv.org/abs/1901.07621
          save_average_policy_memories: saves the collected average_policy memories as a
            tfrecords file in the given location. This is not affected by
            memory_capacity. All memories are saved to disk and not kept in memory
          infer_device: device used for TF-operations in the traversal branch.
            Format is anything accepted by tf.device
          train_device: device used for TF-operations in the NN training steps.
            Format is anything accepted by tf.device
        """
        all_players = list(range(game.num_players()))
        super().__init__(game, all_players)
        self._game = game
        self._save_policy_weights = save_policy_weights
        self._compute_exploitability = True
        self._markov_soccer = markov_soccer
        self._dark_hex = dark_hex
        self._phantom_ttt = phantom_ttt
        self._play_against_random = play_against_random
        self._append_legal_actions_mask = append_legal_actions_mask
        self._num_random_games = 2000
        if self._markov_soccer or self._dark_hex or self._phantom_ttt:
            self._compute_exploitability = False
            self._play_against_random = True
        if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            # `_traverse_game_tree` does not take into account this option.
            raise ValueError('Simulatenous games are not supported.')
        self._batch_size_regret = batch_size_regret
        self._batch_size_value = batch_size_value
        self._batch_size_average_policy = batch_size_average_policy
        self._policy_network_train_steps = policy_network_train_steps
        self._regret_network_train_steps = regret_network_train_steps
        self._value_network_train_steps = value_network_train_steps
        self._policy_network_layers = policy_network_layers
        self._regret_network_layers = regret_network_layers
        self._value_network_layers = value_network_layers
        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()
        self._starting_coins = starting_coins
        self._oshi_zumo = oshi_zumo
        self._battleship = battleship
        if self._oshi_zumo:
            observation_tensor = self._root_node.observation_tensor(0)
            self._embedding_size = len(observation_tensor)
        elif self._battleship:
            observation_tensor = battleship_infostate_tensor(self._root_node, 0)
            self._embedding_size = len(observation_tensor)
        elif self._markov_soccer:
            self._embedding_size = len(self._root_node.observation_tensor(0))
        else:
            self._embedding_size = len(self._root_node.information_state_tensor(0))
        if self._oshi_zumo:
            hist_state = get_oshi_hist_obs(self._root_node.observation_tensor(0), 0, starting_coins, 0)
        elif self._markov_soccer:
            hist_state = get_markov_soccer_hist_obs(self._root_node.observation_tensor(0), 0, 0)
        elif self._battleship:
            hist_state = battleship_history_tensor(self._root_node)
        else:
            hist_state = np.append(self._root_node.information_state_tensor(0),
                                   self._root_node.information_state_tensor(1))
        self._value_embedding_size = len(hist_state)
        self._num_iterations = num_iterations
        self._num_traversals = num_traversals
        self._num_val_fn_traversals = num_val_fn_traversals
        self._reinitialize_regret_networks = reinitialize_regret_networks
        self._reinit_value_network = reinitialize_value_network
        self._num_actions = game.num_distinct_actions()
        self._iteration = 1
        self._learning_rate = learning_rate
        self._save_regret_networks = save_regret_networks
        self._save_average_policy_memories = save_average_policy_memories
        self._infer_device = infer_device
        self._train_device = train_device
        self._memories_tfrecordpath = None
        self._memories_tfrecordfile = None
        self._check_exploitability_every = check_exploitability_every
        self._expl = expl
        self._val_expl = val_expl
        self._importance_sampling = importance_sampling
        self._importance_sampling_threshold = importance_sampling_threshold
        self._clear_value_buffer = clear_value_buffer
        self._nodes_visited = 0
        self._example_info_state = [None, None]
        self._example_hist_state = None
        self._example_legal_actions_mask = [None, None]
        self._squared_errors = []
        self._squared_errors_child = []
        self._balanced_probs = {}
        self._use_balanced_probs = use_balanced_probs
        self._val_op_prob = val_op_prob
        self._val_bootstrap = val_bootstrap
        self._debug_val = debug_val
        self._experiment_string = experiment_string
        self._all_actions = all_actions
        self._random_policy_path = random_policy_path

        # Initialize file save locations
        if self._save_regret_networks:
            os.makedirs(self._save_regret_networks, exist_ok=True)

        if self._save_average_policy_memories:
            if os.path.isdir(self._save_average_policy_memories):
                self._memories_tfrecordpath = os.path.join(
                    self._save_average_policy_memories, 'average_policy_memories.tfrecord')
            else:
                os.makedirs(
                    os.path.split(self._save_average_policy_memories)[0], exist_ok=True)
                self._memories_tfrecordpath = self._save_average_policy_memories

        # Initialize policy network, loss, optimizer
        self._reinitialize_policy_network()

        # Initialize regret networks, losses, optimizers
        self._regret_networks = []
        self._regret_networks_train = []
        self._loss_regrets = []
        self._optimizer_regrets = []
        self._regret_train_step = []
        for player in range(self._num_players):
            with tf.device(self._infer_device):
                self._regret_networks.append(
                    RegretNetwork(self._embedding_size, self._regret_network_layers,
                                  self._num_actions))
            with tf.device(self._train_device):
                self._regret_networks_train.append(
                    RegretNetwork(self._embedding_size,
                                  self._regret_network_layers, self._num_actions))
                self._loss_regrets.append(tf.keras.losses.MeanSquaredError())
                self._optimizer_regrets.append(
                    tf.keras.optimizers.Adam(learning_rate=learning_rate))
                self._regret_train_step.append(self._get_regret_train_graph(player))

        self._create_memories(memory_capacity)

        # Initialize value networks, losses, optimizers
        self._val_network = ValueNetwork(self._value_embedding_size, self._value_network_layers)
        self._val_network_train = ValueNetwork(self._value_embedding_size, self._value_network_layers)
        self._loss_value = tf.keras.losses.MeanSquaredError()
        self._optimizer_value = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._value_train_step = self._get_value_train_graph()
        self._value_test_step = self._get_value_test_graph()

    def _reinitialize_policy_network(self):
        """Reinitalize policy network and optimizer for training."""
        with tf.device(self._train_device):
            self._policy_network = PolicyNetwork(self._embedding_size,
                                                 self._policy_network_layers,
                                                 self._num_actions)
            self._optimizer_policy = tf.keras.optimizers.Adam(
                learning_rate=self._learning_rate)
            self._loss_policy = tf.keras.losses.MeanSquaredError()

    def _reinitialize_regret_network(self, player):
        """Reinitalize player's regret network and optimizer for training."""
        with tf.device(self._train_device):
            self._regret_networks_train[player] = RegretNetwork(
                self._embedding_size, self._regret_network_layers,
                self._num_actions)
            self._optimizer_regrets[player] = tf.keras.optimizers.Adam(
                learning_rate=self._learning_rate)
            self._regret_train_step[player] = (
                self._get_regret_train_graph(player))

    def get_example_info_state(self, player):
        return self._example_info_state[player]

    def get_example_hist_state(self):
        return self._example_hist_state

    def get_example_legal_actions_mask(self, player):
        return self._example_legal_actions_mask[player]

    def _reinitialize_value_network(self):
        """Reinitalize player's value network and optimizer for training."""
        with tf.device(self._train_device):
            self._val_network_train = ValueNetwork(
                self._value_embedding_size, self._value_network_layers)
            self._optimizer_value = tf.keras.optimizers.Adam(
                learning_rate=self._learning_rate)
            self._value_train_step = (self._get_value_train_graph())

    @property
    def regret_buffers(self):
        return self._regret_memories

    @property
    def average_policy_buffer(self):
        return self._average_policy_memories

    def clear_regret_buffers(self):
        for p in range(self._num_players):
            self._regret_memories[p].clear()

    def _create_memories(self, memory_capacity):
        """Create memory buffers and associated feature descriptions."""
        self._average_policy_memories = ReservoirBuffer(memory_capacity)
        self._regret_memories = [
            ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
        ]
        self._value_memory = ReservoirBuffer(memory_capacity)
        self._value_memory_test = ReservoirBuffer(memory_capacity)

        self._average_policy_feature_description = {
            'info_state': tf.io.FixedLenFeature([self._embedding_size], tf.float32),
            'action_probs': tf.io.FixedLenFeature([self._num_actions], tf.float32),
            'iteration': tf.io.FixedLenFeature([1], tf.float32),
            'legal_actions': tf.io.FixedLenFeature([self._num_actions], tf.float32)
        }
        self._regret_feature_description = {
            'info_state': tf.io.FixedLenFeature([self._embedding_size], tf.float32),
            'iteration': tf.io.FixedLenFeature([1], tf.float32),
            'samp_regret': tf.io.FixedLenFeature([self._num_actions], tf.float32),
            'legal_actions': tf.io.FixedLenFeature([self._num_actions], tf.float32)
        }
        self._value_feature_description = {
            'hist_state': tf.io.FixedLenFeature([self._value_embedding_size], tf.float32),
            'iteration': tf.io.FixedLenFeature([1], tf.float32),
            'samp_value': tf.io.FixedLenFeature([1], tf.float32),
            'legal_actions': tf.io.FixedLenFeature([self._num_actions], tf.float32),
        }

    def get_val_weights(self):
        return self._val_network.get_weights()

    def set_val_weights(self, weights):
        self._val_network.set_weights(weights)

    def get_num_calls(self):
        num_calls = 0
        for p in range(self._num_players):
            num_calls += self._regret_memories[p].get_num_calls()
        return num_calls

    def set_iteration(self, iteration):
        self._iteration = iteration

    def get_weights(self):
        regret_weights = [self._regret_networks[player].get_weights() for player in range(self._num_players)]
        return regret_weights

    def get_policy_weights(self):
        policy_weights = self._policy_network.get_weights()
        return policy_weights

    def set_policy_weights(self, policy_weights):
        self._reinitialize_policy_network()
        self._policy_network.set_weights(policy_weights)

    def get_regret_memories(self, player):
        return self._regret_memories[player].get_data()

    def get_value_memory(self):
        return self._value_memory.get_data()

    def clear_value_memory(self):
        self._value_memory.clear()

    def get_value_memory_test(self):
        return self._value_memory_test.get_data()

    def get_average_policy_memories(self):
        return self._average_policy_memories.get_data()

    def get_num_nodes(self):
        return self._nodes_visited

    def get_squared_errors(self):
        return self._squared_errors

    def reset_squared_errors(self):
        self._squared_errors = []

    def get_squared_errors_child(self):
        return self._squared_errors_child

    def reset_squared_errors_child(self):
        self._squared_errors_child = []

    def clear_val_memories_test(self):
        self._value_memory_test.clear()

    def clear_val_memories(self):
        self._value_memory.clear()

    def traverse_game_tree_n_times(self, n, p, train_regret=False, train_value=False,
                                   track_mean_squares=True, on_policy_prob=0., expl=0.6, val_test=False):
        for i in range(n):
            if i > 0:
                track_mean_squares = False
            self._traverse_game_tree(self._root_node, p, my_reach=1.0, opp_reach=1.0, sample_reach=1.0,
                                     my_sample_reach=1.0, train_regret=train_regret, train_value=train_value,
                                     track_mean_squares=track_mean_squares, on_policy_prob=on_policy_prob,
                                     expl=expl, val_test=val_test)

    def init_regret_net(self):
        # initialize regret network
        for p in range(self._num_players):
            example_info_state = self.get_example_info_state(p)
            example_legal_actions_mask = self.get_example_legal_actions_mask(p)
            self.traverse_game_tree_n_times(1, p, track_mean_squares=False)
            self._init_main_regret_network(example_info_state, example_legal_actions_mask, p)

    def init_val_net(self):
        example_hist_state = self.get_example_hist_state()
        example_legal_actions_mask = self.get_example_legal_actions_mask(0)
        self._init_main_val_network(example_hist_state, example_legal_actions_mask)

    def play_game_against_random(self):
        # play one game per player
        reward = 0
        for player in [0, 1]:
            state = self._game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes, probs = zip(*state.chance_outcomes())
                    aidx = np.random.choice(range(len(outcomes)), p=probs)
                    action = outcomes[aidx]
                else:
                    cur_player = state.current_player()
                    legal_actions = state.legal_actions(cur_player)
                    legal_actions_mask = tf.constant(
                        state.legal_actions_mask(cur_player), dtype=tf.float32)
                    obs = tf.constant(state.observation_tensor(), dtype=tf.float32)
                    if len(obs.shape) == 1:
                        obs = tf.expand_dims(obs, axis=0)
                    if cur_player == player:
                        probs = self._policy_network((obs, legal_actions_mask), training=False)
                        probs = probs.numpy()[0]
                        probs /= probs.sum()
                        action = np.random.choice(range(state.num_distinct_actions()), p=probs)
                    elif cur_player == 1 - player:
                        action = random.choice(state.legal_actions())
                    else:
                        print("Got player ", str(cur_player))
                        break
                state.apply_action(action)
            reward += state.returns()[player]
        return reward

    def play_n_games_against_random(self, n):
        total_reward = 0
        for i in range(n):
            reward = self.play_game_against_random()
            total_reward += reward
        return total_reward / (2 * n)

    def print_mse(self):
        # track MSE
        squared_errors = self.get_squared_errors()
        self.reset_squared_errors()
        squared_errors_child = self.get_squared_errors_child()
        self.reset_squared_errors_child()
        print(sum(squared_errors) / len(squared_errors), "Mean Squared Errors")
        print(sum(squared_errors_child) / len(squared_errors_child), "Mean Squared Errors Child")

    def solve(self, save_path_convs=None):
        """Solution logic for Deep CFR."""
        regret_losses = collections.defaultdict(list)
        value_losses = []
        str(datetime.now())
        timestr = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        if self._use_balanced_probs:
            self._get_balanced_probs(self._root_node)

        with tf.device(self._infer_device):
            with contextlib.ExitStack() as stack:
                if self._save_average_policy_memories:
                    self._memories_tfrecordfile = stack.enter_context(
                        tf.io.TFRecordWriter(self._memories_tfrecordpath))
                convs = []
                nodes = []
                self.traverse_game_tree_n_times(1, 0, track_mean_squares=False)

                for i in range(self._num_iterations + 1):
                    print(i)

                    start = time.time()
                    if self._experiment_string is not None:
                        print(self._experiment_string)

                    # init weights
                    self.init_regret_net()
                    self.init_val_net()

                    # train val function
                    self.traverse_game_tree_n_times(self._num_val_fn_traversals, 0,
                                                    train_value=True,
                                                    track_mean_squares=False,
                                                    on_policy_prob=self._val_op_prob,
                                                    expl=self._val_expl)
                    self.traverse_game_tree_n_times(20, 0, train_value=True, track_mean_squares=False,
                                                    on_policy_prob=self._val_op_prob,
                                                    expl=self._val_expl,
                                                    val_test=True)
                    val_traj_time = time.time()
                    print(val_traj_time - start, 'val trajectory time')
                    if self._reinit_value_network:
                        self._reinitialize_value_network()
                    value_losses.append(self._learn_value_network())
                    print(value_losses[-1], 'val loss')
                    test_loss = self._get_value_test_loss()
                    print(test_loss, 'test loss')
                    if self._clear_value_buffer:
                        self.clear_val_memories_test()
                        self.clear_val_memories()
                    val_train_time = time.time()
                    print(val_train_time - val_traj_time, 'val train time')

                    # train regret network
                    if self._oshi_zumo or self._battleship or self._markov_soccer or self._dark_hex or self._phantom_ttt:
                        track_mse = False
                    else:
                        track_mse = True
                    for p in range(self._num_players):
                        regret_start_time = time.time()
                        results = []
                        self.traverse_game_tree_n_times(self._num_traversals, p,
                                                        train_regret=True,
                                                        track_mean_squares=track_mse,
                                                        expl=self._expl)
                        num_nodes = self.get_num_nodes()
                        regret_traj_time = time.time()
                        print(regret_traj_time - regret_start_time, 'regret trajectory time')
                        if self._reinitialize_regret_networks:
                            self._reinitialize_regret_network(p)
                        regret_losses[p].append(self._learn_regret_network(p))
                        if self._save_regret_networks:
                            os.makedirs(self._save_regret_networks, exist_ok=True)
                            self._regret_networks[p].save(
                                os.path.join(self._save_regret_networks,
                                             f'regretnet_p{p}_it{self._iteration:04}'))
                        print(time.time() - regret_traj_time, 'regret train time')
                    if not any([self._oshi_zumo, self._battleship, self._markov_soccer, self._dark_hex,
                                self._phantom_ttt]):
                        self.print_mse()
                    total_regret_time = time.time()
                    print(total_regret_time - val_train_time, 'total regret time')

                    # check exploitability
                    self._iteration += 1
                    if i % self._check_exploitability_every == 0:
                        exp_start_time = time.time()
                        self._reinitialize_policy_network()
                        policy_loss = self._learn_average_policy_network()
                        if self._save_policy_weights:
                            save_path_model = save_path_convs + "/" + timestr
                            os.makedirs(save_path_model, exist_ok=True)
                            model_path = save_path_model + "/policy_nodes_" + str(num_nodes)
                            self._policy_network.save_weights(model_path)
                            print("saved policy to ", model_path)
                            self.save_policy_network(model_path + "full_model")
                            print("saved policy to ", model_path + "full_model")
                        if self._play_against_random:
                            start_time = time.time()
                            avg_reward = self.play_n_games_against_random(self._num_random_games)
                            print(avg_reward, "Average reward against random\n")
                            if self._markov_soccer:
                                nfsp_avg_rew = self._run_n_episodes_against_nfsp(self._num_random_games)
                                print(time.time() - start_time, "eval time")
                                print(nfsp_avg_rew, "Average reward against nfsp\n")
                        if self._compute_exploitability:
                            conv = exploitability.nash_conv(self._game, policy.tabular_policy_from_callable(
                                self._game, self.action_probabilities))
                            convs.append(conv)
                            nodes.append(num_nodes)
                            if save_path_convs:
                                convs_path = save_path_convs + "_convs.npy"
                                nodes_path = save_path_convs + "_nodes.npy"
                                np.save(convs_path, np.array(convs))
                                np.save(nodes_path, np.array(nodes))
                            print(self._iteration, num_nodes, conv)
                            print(time.time() - exp_start_time, 'exp time')

        # Train policy network.
        policy_loss = self._learn_average_policy_network()
        return regret_losses, policy_loss, convs, nodes

    def save_policy_network(self, outputfolder):
        """Saves the policy network to the given folder."""
        os.makedirs(outputfolder, exist_ok=True)
        self._policy_network.save(outputfolder)

    def train_policy_network_from_file(self,
                                       tfrecordpath,
                                       iteration=None,
                                       batch_size_average_policy=None,
                                       policy_network_train_steps=None,
                                       reinitialize_policy_network=True):
        """Trains the policy network from a previously stored tfrecords-file."""
        self._memories_tfrecordpath = tfrecordpath
        if iteration:
            self._iteration = iteration
        if batch_size_average_policy:
            self._batch_size_average_policy = batch_size_average_policy
        if policy_network_train_steps:
            self._policy_network_train_steps = policy_network_train_steps
        if reinitialize_policy_network:
            self._reinitialize_policy_network()
        policy_loss = self._learn_average_policy_network()
        return policy_loss

    def _add_to_average_policy_memory(self, info_state, iteration,
                                      average_policy_action_probs, legal_actions_mask):
        # pylint: disable=g-doc-args
        """Adds the given average_policy data to the memory.

        Uses either a tfrecordsfile on disk if provided, or a reservoir buffer.
        """
        serialized_example = self._serialize_average_policy_memory(
            info_state, iteration, average_policy_action_probs, legal_actions_mask)
        if self._save_average_policy_memories:
            self._memories_tfrecordfile.write(serialized_example)
        else:
            self._average_policy_memories.add(serialized_example)

    def _serialize_average_policy_memory(self, info_state, iteration,
                                         average_policy_action_probs, legal_actions_mask):
        """Create serialized example to store a average_policy entry."""
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'info_state':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=info_state)),
                    'action_probs':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=average_policy_action_probs)),
                    'iteration':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[iteration])),
                    'legal_actions':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=legal_actions_mask))
                }))
        return example.SerializeToString()

    def _deserialize_average_policy_memory(self, serialized):
        """Deserializes a batch of average_policy examples for the train step."""
        tups = tf.io.parse_example(serialized, self._average_policy_feature_description)
        return (tups['info_state'], tups['action_probs'], tups['iteration'],
                tups['legal_actions'])

    def _serialize_regret_memory(self, info_state, iteration, samp_regret,
                                 legal_actions_mask):
        """Create serialized example to store an regret entry."""
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'info_state':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=info_state)),
                    'iteration':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[iteration])),
                    'samp_regret':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=samp_regret)),
                    'legal_actions':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=legal_actions_mask))
                }))
        return example.SerializeToString()

    def _serialize_value_memory(self, hist_state, iteration, samp_value, legal_actions_mask):
        """Create serialized example to store a value entry."""
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'hist_state':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=hist_state)),
                    'iteration':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[iteration])),
                    'samp_value':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[samp_value])),
                    'legal_actions':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=legal_actions_mask))
                }))
        return example.SerializeToString()

    def _deserialize_regret_memory(self, serialized):
        """Deserializes a batch of regret examples for the train step."""
        tups = tf.io.parse_example(serialized, self._regret_feature_description)
        return (tups['info_state'], tups['samp_regret'], tups['iteration'],
                tups['legal_actions'])

    def _deserialize_value_memory(self, serialized):
        """Deserializes a batch of regret examples for the train step."""
        tups = tf.io.parse_example(serialized, self._value_feature_description)
        return (tups['hist_state'], tups['samp_value'], tups['iteration'], tups['legal_actions'])

    def _baseline(self, state, aidx):  # pylint: disable=unused-argument
        # Default to vanilla outcome sampling
        return 0

    def _baseline_corrected_child_value(self, state, sampled_aidx,
                                        aidx, child_value, sample_prob):
        # Applies Eq. 9 of Schmid et al. '19
        baseline = self._baseline(state, aidx)
        if aidx == sampled_aidx:
            return baseline + (child_value - baseline) / sample_prob
        else:
            return baseline

    def _exact_value(self, state, update_player):
        state = state.clone()
        if state.is_terminal():
            return state.player_return(update_player)
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            val = 0
            for aidx in range(len(outcomes)):
                new_state = state.child(outcomes[aidx])
                val += probs[aidx] * self._exact_value(new_state, update_player)
            return val
        cur_player = state.current_player()
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)
        _, policy = self._sample_action_from_regret(state, cur_player)
        val = 0
        for aidx in range(num_legal_actions):
            new_state = state.child(legal_actions[aidx])
            val += policy[aidx] * self._exact_value(new_state, update_player)
        return val

    def _get_balanced_probs(self, state):
        if state.is_terminal():
            return 1
        elif state.is_chance_node():
            legal_actions = state.legal_actions()
            num_nodes = 0
            for action in legal_actions:
                num_nodes += self._get_balanced_probs(state.child(action))
            return num_nodes
        else:
            legal_actions = state.legal_actions()
            num_nodes = 0
            balanced_probs = np.zeros((state.num_distinct_actions()))
            for action in legal_actions:
                nodes = self._get_balanced_probs(state.child(action))
                balanced_probs[action] = nodes
                num_nodes += nodes
            self._balanced_probs[state.information_state_string()] = balanced_probs / balanced_probs.sum()
            return num_nodes

    def _traverse_game_tree(self, state, player, my_reach, opp_reach, sample_reach,
                            my_sample_reach, train_regret, train_value,
                            on_policy_prob=0., track_mean_squares=True, expl=1.0, val_test=False, last_action=0):
        """Performs a traversal of the game tree using external sampling.

        Over a traversal the regret and average_policy memories are populated with
        computed regret values and matched regrets respectively if train_regret=True.
        If train_value=True then we use traversals to train the history value function.

        Args:
          state: Current OpenSpiel game state.
          player: (int) Player index for this traversal.

        Returns:
          Recursively returns expected payoffs for each action.
        """
        self._nodes_visited += 1
        if state.is_terminal():
            # Terminal state get returns.
            return state.returns()[player], state.returns()[player]
        elif state.is_chance_node():
            # If this is a chance node, sample an action
            outcomes, probs = zip(*state.chance_outcomes())
            aidx = np.random.choice(range(len(outcomes)), p=probs)
            action = outcomes[aidx]
            new_state = state.child(action)
            return self._traverse_game_tree(new_state, player, my_reach,
                                            probs[aidx] * opp_reach, probs[aidx] * sample_reach, my_sample_reach,
                                            train_regret, train_value, expl=expl,
                                            track_mean_squares=track_mean_squares, val_test=val_test,
                                            last_action=action)

        # with probability equal to op_prob, we switch over to on-policy rollout for remainder of trajectory
        # used for value estimation to get coverage but not needing importance sampling
        if expl != 0.:
            if np.random.rand() < on_policy_prob:
                expl = 0.

        cur_player = state.current_player()
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)
        num_actions = state.num_distinct_actions()
        _, policy = self._sample_action_from_regret(state, state.current_player())

        if cur_player == player or train_value:
            uniform_policy = (np.array(state.legal_actions_mask()) / num_legal_actions)
            if self._use_balanced_probs:
                uniform_policy = self._balanced_probs[state.information_state_string()]
            sample_policy = expl * uniform_policy + (1.0 - expl) * policy
        else:
            sample_policy = policy

        sample_policy /= sample_policy.sum()
        sampled_action = np.random.choice(range(state.num_distinct_actions()), p=sample_policy)
        orig_state = state.clone()
        new_state = state.child(sampled_action)

        child_value = self._estimate_value_from_hist(new_state.clone(), player, last_action=sampled_action)
        value_estimate = self._estimate_value_from_hist(state.clone(), player, last_action=last_action)

        if track_mean_squares:
            oracle_child_value = self._exact_value(new_state.clone(), player)
            oracle_value_estimate = self._exact_value(state.clone(), player)
            squared_error = (oracle_value_estimate - value_estimate) ** 2
            self._squared_errors.append(squared_error)
            squared_child_error = (oracle_child_value - child_value) ** 2
            self._squared_errors_child.append(squared_child_error)

        if cur_player == player:
            new_my_reach = my_reach * policy[sampled_action]
            new_opp_reach = opp_reach
            new_my_sample_reach = my_sample_reach * sample_policy[sampled_action]
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_action]
            new_my_sample_reach = my_sample_reach
        new_sample_reach = sample_reach * sample_policy[sampled_action]

        iw_sampled_value, sampled_value = self._traverse_game_tree(new_state, player, new_my_reach,
                                                                   new_opp_reach, new_sample_reach, new_my_sample_reach,
                                                                   train_regret, train_value, expl=expl,
                                                                   track_mean_squares=track_mean_squares,
                                                                   val_test=val_test, last_action=sampled_action)
        importance_weighted_sampled_value = iw_sampled_value * policy[sampled_action] / sample_policy[sampled_action]

        # Compute each of the child estimated values.
        child_values = np.zeros(num_actions, dtype=np.float64)
        if self._all_actions:
            for aidx in range(num_legal_actions):
                cloned_state = orig_state.clone()
                action = legal_actions[aidx]
                new_cloned_state = cloned_state.child(action)
                child_values[action] = self._estimate_value_from_hist(new_cloned_state.clone(), player,
                                                                      last_action=action)
        else:
            child_values[sampled_action] = child_value / sample_policy[sampled_action]

        if train_regret:
            if cur_player == player:
                cf_action_values = 0 * policy
                for action in range(num_actions):
                    if self._importance_sampling:
                        action_sample_reach = my_sample_reach * sample_policy[sampled_action]
                        cf_value = value_estimate * min(1 / my_sample_reach, self._importance_sampling_threshold)
                        cf_action_value = child_values[action] * min(1 / action_sample_reach,
                                                                     self._importance_sampling_threshold)
                    else:
                        cf_action_value = child_values[action]
                        cf_value = value_estimate
                    cf_action_values[action] = cf_action_value

                samp_regret = (cf_action_values - cf_value) * state.legal_actions_mask(player)
                if self._oshi_zumo:
                    network_input = state.observation_tensor()
                elif self._battleship:
                    network_input = battleship_infostate_tensor(state, cur_player)
                elif self._markov_soccer:
                    network_input = state.observation_tensor()
                else:
                    network_input = state.information_state_tensor()

                self._regret_memories[player].add(self._serialize_regret_memory(network_input,
                                                                                self._iteration,
                                                                                samp_regret,
                                                                                state.legal_actions_mask(
                                                                                    player)))
            else:
                if self._oshi_zumo:
                    obs_input = state.observation_tensor(cur_player)
                elif self._battleship:
                    obs_input = battleship_infostate_tensor(state, cur_player)
                elif self._markov_soccer:
                    obs_input = state.observation_tensor(cur_player)
                else:
                    obs_input = state.information_state_tensor(cur_player)

                self._add_to_average_policy_memory(obs_input, self._iteration,
                                                   policy, state.legal_actions_mask(cur_player))

        # value function predicts value for player 0
        if train_value:
            # if op_prob = 0 then we are doing importance weighted sampling
            # if op_prob > 0 then we need to wait until expl = 0 to get pure on-policy rollouts
            if on_policy_prob == 0 or expl == 0:
                if self._oshi_zumo:
                    hist_state = get_oshi_hist_obs(state.observation_tensor(0), cur_player, self._starting_coins,
                                                   last_action)
                elif self._battleship:
                    hist_state = battleship_history_tensor(state)
                elif self._markov_soccer:
                    hist_state = get_markov_soccer_hist_obs(state.observation_tensor(0), cur_player, last_action)
                else:
                    hist_state = np.append(state.information_state_tensor(0), state.information_state_tensor(1))

                assert player == 0
                if self._val_bootstrap:
                    if self._all_actions:
                        target = policy @ child_values
                    else:
                        target = child_value * policy[sampled_action] / sample_policy[sampled_action]
                elif self._debug_val:
                    target = child_value * policy[sampled_action] / sample_policy[sampled_action]
                    print(target, 'value target')
                else:
                    target = iw_sampled_value
                if val_test:
                    self._value_memory_test.add(
                        self._serialize_value_memory(hist_state, self._iteration, target,
                                                     state.legal_actions_mask(cur_player)))
                else:
                    self._value_memory.add(
                        self._serialize_value_memory(hist_state, self._iteration, target,
                                                     state.legal_actions_mask(cur_player)))

        return importance_weighted_sampled_value, sampled_value

    @tf.function
    def _init_main_regret_network(self, info_state, legal_actions_mask, player):
        """TF-Graph to calculate regret matching."""
        regrets = self._regret_networks[player](
            (tf.expand_dims(info_state, axis=0), legal_actions_mask),
            training=False)[0]

    @tf.function
    def _init_main_val_network(self, hist_state, legal_actions_mask):
        """TF-Graph to calculate regret matching."""
        estimated_val = \
            self._val_network((tf.expand_dims(hist_state, axis=0), legal_actions_mask), training=False)[0]

    @tf.function
    def _get_matched_regrets(self, info_state, legal_actions_mask, player):
        """TF-Graph to calculate regret matching."""
        regrets = self._regret_networks[player](
            (tf.expand_dims(info_state, axis=0), legal_actions_mask),
            training=False)[0]

        regrets = tf.maximum(regrets, 0)
        summed_regret = tf.reduce_sum(regrets)
        if summed_regret > 0:
            matched_regrets = regrets / summed_regret
        else:
            matched_regrets = tf.one_hot(
                tf.argmax(tf.where(legal_actions_mask == 1, regrets, -10e20)),
                self._num_actions)
        return regrets, matched_regrets

    @tf.function
    def _get_estimated_value(self, hist_state, legal_actions_mask):
        """TF-Graph to calculate regret matching."""
        estimated_val = \
            self._val_network((tf.expand_dims(hist_state, axis=0), legal_actions_mask), training=False)[0]
        return estimated_val

    def _sample_action_from_regret(self, state, player):
        """Returns an info state policy by applying regret-matching.

        Args:
          state: Current OpenSpiel game state.
          player: (int) Player index over which to compute regrets.

        Returns:
          1. (np-array) regret values for info state actions indexed by action.
          2. (np-array) Matched regrets, prob for actions indexed by action.
        """
        if self._oshi_zumo:
            observation_tensor = state.observation_tensor(player)
            info_state = tf.constant(observation_tensor, dtype=tf.float32)
        elif self._battleship:
            info_state = tf.constant(battleship_infostate_tensor(state, player),
                                     dtype=tf.float32)
        elif self._markov_soccer:
            info_state = tf.constant(state.observation_tensor(player), dtype=tf.float32)
        else:
            info_state = tf.constant(
                state.information_state_tensor(player), dtype=tf.float32)
        legal_actions_mask = tf.constant(
            state.legal_actions_mask(player), dtype=tf.float32)
        self._example_info_state[player] = info_state
        self._example_legal_actions_mask[player] = legal_actions_mask
        regrets, matched_regrets = self._get_matched_regrets(
            info_state, legal_actions_mask, player)
        return regrets.numpy(), matched_regrets.numpy()

    def _estimate_value_from_hist(self, state, player, last_action=0):
        """Returns an info state policy by applying regret-matching.

        Args:
          state: Current OpenSpiel game state.
          player: (int) Player index over which to compute regrets.

        Returns:
          1. (np-array) regret values for info state actions indexed by action.
          2. (np-array) Matched regrets, prob for actions indexed by action.
        """
        state = state.clone()
        if state.is_terminal():
            return state.player_return(player)

        if self._oshi_zumo:
            hist_state = get_oshi_hist_obs(state.observation_tensor(0), state.current_player(), self._starting_coins,
                                           last_action)
        elif self._battleship:
            hist_state = battleship_history_tensor(state)
        elif self._markov_soccer:
            hist_state = get_markov_soccer_hist_obs(state.observation_tensor(0), state.current_player(), last_action)
        else:
            hist_state = np.append(state.information_state_tensor(0), state.information_state_tensor(1))

        self._example_hist_state = hist_state
        hist_state = tf.constant(hist_state, dtype=tf.float32)
        legal_actions_mask = tf.constant(
            state.legal_actions_mask(player), dtype=tf.float32)
        estimated_value = self._get_estimated_value(hist_state, legal_actions_mask)
        if player == 1:
            estimated_value = -estimated_value
        return estimated_value.numpy()

    def action_probabilities(self, state):
        """Returns action probabilities dict for a single batch."""
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        legal_actions_mask = tf.constant(
            state.legal_actions_mask(cur_player), dtype=tf.float32)
        if self._oshi_zumo:
            observation_tensor = state.observation_tensor()
            info_state_vector = tf.constant(observation_tensor, dtype=tf.float32)
        elif self._battleship:
            info_state_vector = tf.constant(battleship_infostate_tensor(state, cur_player),
                                            dtype=tf.float32)
        elif self._markov_soccer:
            info_state_vector = tf.constant(state.observation_tensor(), dtype=tf.float32)
        else:
            info_state_vector = tf.constant(
                state.information_state_tensor(), dtype=tf.float32)
        if len(info_state_vector.shape) == 1:
            info_state_vector = tf.expand_dims(info_state_vector, axis=0)
        probs = self._policy_network((info_state_vector, legal_actions_mask),
                                     training=False)
        probs = probs.numpy()
        return {action: probs[0][action] for action in legal_actions}

    def _get_regret_dataset(self, player):
        """Returns the collected regrets for the given player as a dataset."""
        data = self.get_regret_memories(player)
        data = tf.data.Dataset.from_tensor_slices(data)
        data = data.shuffle(REGRET_TRAIN_SHUFFLE_SIZE)
        data = data.repeat()
        data = data.batch(self._batch_size_regret)
        data = data.map(self._deserialize_regret_memory)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data

    def _get_value_dataset(self):
        """Returns the collected value estimates for the given player as a dataset."""
        data = self.get_value_memory()
        data = tf.data.Dataset.from_tensor_slices(data)
        data = data.shuffle(VALUE_TRAIN_SHUFFLE_SIZE)
        data = data.repeat()
        data = data.batch(self._batch_size_value)
        data = data.map(self._deserialize_value_memory)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data

    def _get_value_dataset_test(self):
        """Returns the collected value estimates for the given player as a dataset."""
        data = self.get_value_memory_test()
        data = tf.data.Dataset.from_tensor_slices(data)
        data = data.shuffle(VALUE_TRAIN_SHUFFLE_SIZE)
        data = data.repeat()
        data = data.batch(self._batch_size_value)
        data = data.map(self._deserialize_value_memory)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data

    def _get_value_test_loss(self):
        with tf.device(self._train_device):
            tfit = tf.constant(self._iteration, dtype=tf.float32)
            data = self._get_value_dataset_test()
            for d in data.take(1):
                main_loss = self._value_test_step(*d, tfit)
                if self._debug_val:
                    print(main_loss, 'test loss')
        return main_loss

    def _get_regret_train_graph(self, player):
        """Return TF-Graph to perform regret network train step."""

        @tf.function
        def train_step(info_states, regrets, iterations, masks, iteration):
            model = self._regret_networks_train[player]
            with tf.GradientTape() as tape:
                preds = model((info_states, masks), training=True)
                main_loss = self._loss_regrets[player](regrets, preds, sample_weight=iterations * 2 / iteration)
                loss = tf.add_n([main_loss], model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            self._optimizer_regrets[player].apply_gradients(
                zip(gradients, model.trainable_variables))

            return main_loss

        return train_step

    def _get_value_train_graph(self):
        """Return TF-Graph to perform value network train step."""

        @tf.function
        def train_step(full_hist_states, values, iterations, masks, iteration):
            model = self._val_network_train
            with tf.GradientTape() as tape:
                preds = model((full_hist_states, masks), training=True)
                main_loss = self._loss_value(
                    values, preds, sample_weight=1)
                loss = tf.add_n([main_loss], model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            self._optimizer_value.apply_gradients(
                zip(gradients, model.trainable_variables))
            return main_loss

        return train_step

    def _get_value_test_graph(self):
        """Return TF-Graph to perform value network train step."""

        @tf.function
        def test_step(full_hist_states, values, iterations, masks, iteration):
            model = self._val_network
            with tf.GradientTape() as tape:
                preds = model((full_hist_states, masks), training=True)
                main_loss = self._loss_value(
                    values, preds, sample_weight=1)
                loss = tf.add_n([main_loss], model.losses)
            return main_loss

        return test_step

    def _learn_value_network(self):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Args:
          player: (int) player index.

        Returns:
          The average loss over the regret network of the last batch.
        """

        with tf.device(self._train_device):
            tfit = tf.constant(self._iteration, dtype=tf.float32)
            data = self._get_value_dataset()
            for d in data.take(self._value_network_train_steps):
                main_loss = self._value_train_step(*d, tfit)
                if self._debug_val:
                    print(main_loss, 'main val loss')

        self._val_network.set_weights(
            self._val_network_train.get_weights())
        return main_loss

    def _learn_regret_network(self, player):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Args:
          player: (int) player index.

        Returns:
          The average loss over the regret network of the last batch.
        """

        with tf.device(self._train_device):
            tfit = tf.constant(self._iteration, dtype=tf.float32)
            data = self._get_regret_dataset(player)
            for d in data.take(self._regret_network_train_steps):
                main_loss = self._regret_train_step[player](*d, tfit)

        self._regret_networks[player].set_weights(
            self._regret_networks_train[player].get_weights())
        return main_loss

    def _get_average_policy_dataset(self):
        """Returns the collected average_policy memories as a dataset."""
        if self._memories_tfrecordpath:
            data = tf.data.TFRecordDataset(self._memories_tfrecordpath)
        else:
            data = self.get_average_policy_memories()
            data = tf.data.Dataset.from_tensor_slices(data)
        data = data.shuffle(AVERAGE_POLICY_TRAIN_SHUFFLE_SIZE)
        data = data.repeat()
        data = data.batch(self._batch_size_average_policy)
        data = data.map(self._deserialize_average_policy_memory)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data

    def _learn_average_policy_network(self):
        """Compute the loss over the average_policy network.

        Returns:
          The average loss obtained on the last training batch of transitions
          or `None`.
        """

        @tf.function
        def train_step(info_states, action_probs, iterations, masks):
            model = self._policy_network
            with tf.GradientTape() as tape:
                preds = model((info_states, masks), training=True)
                main_loss = self._loss_policy(
                    action_probs, preds, sample_weight=iterations * 2 / self._iteration)
                loss = tf.add_n([main_loss], model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            self._optimizer_policy.apply_gradients(
                zip(gradients, model.trainable_variables))
            return main_loss

        with tf.device(self._train_device):
            data = self._get_average_policy_dataset()
            for d in data.take(self._policy_network_train_steps):
                main_loss = train_step(*d)

        return main_loss


if __name__ == "__main__":
    # Quick example how to run on Kuhn
    # Hyperparameters not tuned

    train_device = 'cpu'
    save_path = "./tmp/results/"
    os.makedirs(save_path, exist_ok=True)

    game = pyspiel.load_game("kuhn_poker")

    iters = 30
    num_traversals = 500
    num_val_fn_traversals = 500
    regret_train_steps = 200
    val_train_steps = 200
    policy_net_train_steps = 1000
    batch_size_regret = 256
    batch_size_val = 256

    deep_cfr_solver = ESCHERSolver(
        game,
        num_traversals=int(num_traversals),
        num_iterations=iters,
        check_exploitability_every=10,
        compute_exploitability=True,
        regret_network_train_steps=regret_train_steps,
        policy_network_train_steps=policy_net_train_steps,
        batch_size_regret=batch_size_regret,
        value_network_train_steps=val_train_steps,
        batch_size_value=batch_size_val,
        train_device=train_device,
    )

    regret, pol_loss, convs, nodes = deep_cfr_solver.solve(save_path_convs=save_path)
