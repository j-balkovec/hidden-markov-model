# Hidden Markov Model Implementation

## Module: src.py
- **Version:** 1.0
- **Author:** Jakob Balkovec
- **Date:** 15-Nov-2023

## Description
This module contains the implementation of a Hidden Markov Model class, designed to model systems with hidden states and observable outputs. The class provides methods for computing forward and backward probabilities of observation sequences, running the expectation-maximization algorithm to learn model parameters, and predicting the most likely sequence of hidden states given an observation sequence.

## Classes
### HiddenMarkovModel
- A class representing a Hidden Markov Model.

### Methods
1. **`forward_algorithm(observation_sequence: List[str]) -> np.ndarray`**
   - Computes the forward probabilities for a given observation sequence.

2. **`backward_algorithm(observation_sequence: List[str]) -> np.ndarray`**
   - Computes the backward probabilities for a given observation sequence.

3. **`expectation_maximization_algorithm(observation_sequence: List[str], iterations: int = 100) -> 'HiddenMarkovModel'`**
   - Runs the expectation-maximization algorithm to learn the model parameters from a given observation sequence.

4. **`predict(observation_sequence: List[str]) -> List[str]`**
   - Predicts the most likely sequence of states given an observation sequence.

5. **`json_dumps() -> bool`**
   - Exports the model parameters into a JSON file.

### Attributes
- **`states` (List[str])**: A list of all possible states in the model.
- **`observations` (List[str])**: A list of all possible observations in the model.
- **`transition_probabilities` (Dict[str, Dict[str, float]])**: A dictionary of dictionaries representing the transition probabilities between states.
- **`emission_probabilities` (Dict[str, Dict[str, float]])**: A dictionary of dictionaries representing the emission probabilities of observations given states.
- **`initial_probabilities` (Dict[str, float])**: A dictionary representing the initial probabilities of each state.
- **`predicted_states` (List[str])**: The list of predicted states.

## Usage
```python
###
Define states, observations, ... in `constants.py`
###
from constants import *

# Run the provided main function
hmm_main(STATES_2_, OBSERVATIONS_2_, TRANSITION_PROB_2_, EMISSION_PROB_2_, INITIAL_PROB_2_, OBSERVATION_SEQ_2_)
```

## Exported JSON Format
The `json_dumps` method exports the following data into a JSON file named `model_parameters.json`:
```JSON
{
  "states": [...],
  "observations": [...],
  "transition_probabilities": {...},
  "emission_probabilities": {...},
  "initial_probabilities": {...},
  "predicted_states": [...]
}
```

## Main Function

The `hmm_main()` function demonstrates the usage of the `HiddenMarkovModel` class, including running the expectation-maximization algorithm and exporting model parameters to a `JSON` file.

To run the main function:
```python
hmm_main()
```

## License
This Hidden Markov Model Implementation is licensed under the [MIT License].
