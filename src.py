import numpy as np
from typing import List, Dict

STATES_ = ['Sunny', 'Rainy']
OBSERVATIONS_ = ['Dry', 'Wet']

TRANSITION_PROB_ = {
  'Sunny': {'Sunny': 0.7, 'Rainy': 0.3},
  'Rainy': {'Sunny': 0.4, 'Rainy': 0.6}
}
EMISSION_PROB_ = {
  'Sunny': {'Dry': 0.9, 'Wet': 0.1},
  'Rainy': {'Dry': 0.3, 'Wet': 0.7}
}

INITIAL_PROB_ = {'Sunny': 0.5, 'Rainy': 0.5}
OBSERVATION_SEQ = ['Dry', 'Wet', 'Dry', 'Dry', 'Wet']



class HiddenMarkovModel:
  def __init__(self,
               states: List[str], 
               observations: List[str], 
               transition_probabilities: Dict[str, Dict[str, float]], 
               emission_probabilities: Dict[str, Dict[str, float]], 
               initial_probabilities: Dict[str, float]) -> None:
    
    self.states = states
    self.observations = observations
    self.transition_probabilities = transition_probabilities
    self.emission_probabilities = emission_probabilities
    self.initial_probabilities = initial_probabilities

  def forward_algorithm(self, observation_sequence: List[str]) -> np.ndarray:
    alpha = np.zeros((len(observation_sequence), len(self.states)))
    alpha[0] = [self.initial_probabilities[state] * self.emission_probabilities[state][observation_sequence[0]] for state in self.states]
    for t in range(1, len(observation_sequence)):
      for j in range(len(self.states)):
        alpha[t][j] = sum(alpha[t-1][i] * self.transition_probabilities[self.states[i]][self.states[j]] * self.emission_probabilities[self.states[j]][observation_sequence[t]] for i in range(len(self.states)))
    return alpha

  def backward_algorithm(self, observation_sequence: List[str]) -> np.ndarray:
    beta = np.zeros((len(observation_sequence), len(self.states)))
    beta[-1] = [1] * len(self.states)
    for t in range(len(observation_sequence)-2, -1, -1):
      for i in range(len(self.states)):
        beta[t][i] = sum(self.transition_probabilities[self.states[i]][self.states[j]] * self.emission_probabilities[self.states[j]][observation_sequence[t+1]] * beta[t+1][j] for j in range(len(self.states)))
    return beta

  def expectation_maximization_algorithm(self, observation_sequence: List[str], iterations: int = 100) -> 'HiddenMarkovModel':
    alpha = self.forward_algorithm(observation_sequence)
    beta = self.backward_algorithm(observation_sequence)
    gamma = np.zeros((len(observation_sequence), len(self.states)))
    xi = np.zeros((len(observation_sequence)-1, len(self.states), len(self.states)))
    for t in range(len(observation_sequence)):
      gamma[t] = alpha[t] * beta[t] / sum(alpha[-1])
      if t == len(observation_sequence) - 1:
        continue
      for i in range(len(self.states)):
        for j in range(len(self.states)):
          xi[t][i][j] = alpha[t][i] * self.transition_probabilities[self.states[i]][self.states[j]] * self.emission_probabilities[self.states[j]][observation_sequence[t+1]] * beta[t+1][j] / sum(alpha[-1])
    for iteration in range(iterations):
      for i in range(len(self.states)):
        self.initial_probabilities[self.states[i]] = gamma[0][i]
        for j in range(len(self.states)):
          self.transition_probabilities[self.states[i]][self.states[j]] = sum(xi[t][i][j] for t in range(len(observation_sequence)-1)) / sum(gamma[t][i] for t in range(len(observation_sequence)-1))
        for k in range(len(self.observations)):
          self.emission_probabilities[self.states[i]][self.observations[k]] = sum(gamma[t][i] for t in range(len(observation_sequence)) if observation_sequence[t] == self.observations[k]) / sum(gamma[t][i] for t in range(len(observation_sequence)))
    return self

  def predict(self, observation_sequence: List[str]) -> List[str]:
    alpha = self.forward_algorithm(observation_sequence)
    return [self.states[i] for i in np.argmax(alpha, axis=1)]

def main() -> None:
    hmm = HiddenMarkovModel(STATES_, OBSERVATIONS_, TRANSITION_PROB_, EMISSION_PROB_, INITIAL_PROB_)
    hmm.expectation_maximization_algorithm(OBSERVATION_SEQ)
    predicted_states = hmm.predict(OBSERVATION_SEQ)
    print("\nThe predicted states are: ", predicted_states)

if __name__ == "__main__":
    main()
