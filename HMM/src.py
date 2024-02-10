"""
Module: src.py
Version: 1.0
Author: Jakob Balkovec
Date: 15-Nov-2023

Description:
This module contains the implementation of a Hidden Markov Model class,
which can be used to model a system with hidden states and observable outputs.
The class provides methods for computing the forward and backward probabilities
of a given observation sequence, as well as for running the 
expectation-maximization algorithm to learn the model parameters 
from a given observation sequence. The class also provides a method 
for predicting the most likely sequence of hidden states given an 
observation sequence.

Classes:
- HiddenMarkovModel: A class representing a Hidden Markov Model.

"""

"""__imports__"""
from typing import List, Dict
import numpy as np
import json
from constants import *
class HiddenMarkovModel:
    """
    A class representing a Hidden Markov Model.

    Attributes:
      states (List[str]): A list of all possible states in the model.
      observations (List[str]): A list of all possible observations in the model.
      transition_probabilities (Dict[str, Dict[str, float]]): A dictionary of dictionaries representing the transition probabilities between states.
      emission_probabilities (Dict[str, Dict[str, float]]): A dictionary of dictionaries representing the emission probabilities of observations given states.
      initial_probabilities (Dict[str, float]): A dictionary representing the initial probabilities of each state.

    Methods:
      forward_algorithm(observation_sequence: List[str]) -> np.ndarray:
        Computes the forward probabilities for a given observation sequence.
      backward_algorithm(observation_sequence: List[str]) -> np.ndarray:
        Computes the backward probabilities for a given observation sequence.
      expectation_maximization_algorithm(observation_sequence: List[str], iterations: int = 100) -> 'HiddenMarkovModel':
        Runs the expectation-maximization algorithm to learn the model parameters from a given observation sequence.
      predict(observation_sequence: List[str]) -> List[str]:
        Predicts the most likely sequence of states given an observation sequence.
    """

    def __init__(self,
                 states: List[str],
                 observations: List[str],
                 transition_probabilities: Dict[str, Dict[str, float]],
                 emission_probabilities: Dict[str, Dict[str, float]],
                 initial_probabilities: Dict[str, float],
                 predicted_states: List[str]) -> None:
        """
        Initializes a HiddenMarkovModel object.

        Args:
          states (List[str]): A list of all possible states in the model.
          observations (List[str]): A list of all possible observations in the model.
          transition_probabilities (Dict[str, Dict[str, float]]): A dictionary of dictionaries representing the transition probabilities between states.
          emission_probabilities (Dict[str, Dict[str, float]]): A dictionary of dictionaries representing the emission probabilities of observations given states.
          initial_probabilities (Dict[str, float]): A dictionary representing the initial probabilities of each state.
        """
        self.states = states
        self.observations = observations
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities
        self.initial_probabilities = initial_probabilities
        self.predicted_states = predicted_states

    def forward_algorithm(self, observation_sequence: List[str]) -> np.ndarray:
        """
        Computes the forward probabilities for a given observation sequence.

        Args:
          observation_sequence (List[str]): The observation sequence.

        Returns:
          np.ndarray: A 2D numpy array representing the forward probabilities.
        """
        alpha = np.zeros((len(observation_sequence), len(self.states)))
        alpha[0] = [self.initial_probabilities[state] * self.emission_probabilities[state].get(
            observation_sequence[0], 0.0) for state in self.states]
        for t in range(1, len(observation_sequence)):
            for j in range(len(self.states)):
                alpha[t][j] = sum(alpha[t-1][i] * self.transition_probabilities[self.states[i]][self.states[j]] *
                                  self.emission_probabilities[self.states[j]][observation_sequence[t]] for i in range(len(self.states)))
        return alpha

    def backward_algorithm(self, observation_sequence: List[str]) -> np.ndarray:
        """
        Computes the backward probabilities for a given observation sequence.

        Args:
          observation_sequence (List[str]): The observation sequence.

        Returns:
          np.ndarray: A 2D numpy array representing the backward probabilities.
        """
        beta = np.zeros((len(observation_sequence), len(self.states)))
        beta[-1] = [1] * len(self.states)
        for t in range(len(observation_sequence)-2, -1, -1):
            for i in range(len(self.states)):
                beta[t][i] = sum(self.transition_probabilities[self.states[i]][self.states[j]] * self.emission_probabilities[self.states[j]]
                                 [observation_sequence[t+1]] * beta[t+1][j] for j in range(len(self.states)))
        return beta

    def expectation_maximization_algorithm(self, observation_sequence: List[str], iterations: int = 100) -> 'HiddenMarkovModel':
        """
        Runs the expectation-maximization algorithm to learn the model parameters from a given observation sequence.

        Args:
          observation_sequence (List[str]): The observation sequence.
          iterations (int): The number of iterations to run the algorithm.

        Returns:
          HiddenMarkovModel: The updated HiddenMarkovModel object.
        """
        alpha = self.forward_algorithm(observation_sequence)
        beta = self.backward_algorithm(observation_sequence)
        gamma = np.zeros((len(observation_sequence), len(self.states)))
        xi = np.zeros((len(observation_sequence)-1,
                      len(self.states), len(self.states)))
        for t in range(len(observation_sequence)):
            gamma[t] = alpha[t] * beta[t] / sum(alpha[-1])
            if t == len(observation_sequence) - 1:
                continue
            for i in range(len(self.states)):
                for j in range(len(self.states)):
                    xi[t][i][j] = alpha[t][i] * self.transition_probabilities[self.states[i]][self.states[j]] * \
                        self.emission_probabilities[self.states[j]
                                                    ][observation_sequence[t+1]] * beta[t+1][j] / sum(alpha[-1])
        for iteration in range(iterations):
            for i in range(len(self.states)):
                self.initial_probabilities[self.states[i]] = gamma[0][i]
                for j in range(len(self.states)):
                    self.transition_probabilities[self.states[i]][self.states[j]] = sum(xi[t][i][j] for t in range(
                        len(observation_sequence)-1)) / sum(gamma[t][i] for t in range(len(observation_sequence)-1))
                for k in range(len(self.observations)):
                    self.emission_probabilities[self.states[i]][self.observations[k]] = sum(gamma[t][i] for t in range(len(
                        observation_sequence)) if observation_sequence[t] == self.observations[k]) / sum(gamma[t][i] for t in range(len(observation_sequence)))
        return self

    def predict(self, observation_sequence: List[str]) -> None:
        """
        Predicts the most likely sequence of states given an observation sequence.

        Args:
          observation_sequence (List[str]): The observation sequence.

        Returns:
          List[str]: A list of the most likely states.
        """
        alpha = self.forward_algorithm(observation_sequence)
        self.predicted_states = [self.states[i]
                                 for i in np.argmax(alpha, axis=1)]

    def json_dumps(self) -> bool:
        """
        Exports the data into a JSON file.
        """
        data = {
            "states": self.states,
            "observations": self.observations,
            "transition_probabilities": self.transition_probabilities,
            "emission_probabilities": self.emission_probabilities,
            "initial_probabilities": self.initial_probabilities,
            "predicted_states": self.predicted_states
        }
        try:
            with open('model_parameters.json', 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            # Assume
            print("[Error]: Could not export model parameters to JSON file.")
            return False
        finally:
            return True


def hmm_main(states: list[str],
             observations: list[str],
             transition_prob: dict[str, dict[str, float]],
             emission_prob: dict[str, dict[str, float]],
             initial_prob: dict[str, float],
             observation_seq: list[str]) -> bool:
    predicted_states = None
    hmm = HiddenMarkovModel(states, observations, transition_prob,
                            emission_prob, initial_prob, predicted_states)
    hmm.expectation_maximization_algorithm(observation_seq)
    hmm.predict(observation_seq)

    if(hmm.json_dumps() == True):
        print(
            "\n{\n\n[Hidden Markov Model]\n[SUCCESS]: Model parameters exported to JSON file.\n\n}\n")
        return True
    print(
        "\n{\n\n[Hidden Markov Model]\n[FAILURE]: Something went wrong.\n\n}\n")
    return False
