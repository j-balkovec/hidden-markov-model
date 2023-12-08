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
OBSERVATION_SEQ_ = ['Dry', 'Wet', 'Dry', 'Dry', 'Wet']

STATES_2_ = ['Healthy', 'Fever', 'Dizzy']
OBSERVATIONS_2_ = ['Normal', 'Cold', 'Hot']

TRANSITION_PROB_2_ = {
  'Healthy': {'Healthy': 0.4, 'Fever': 0.3, 'Dizzy': 0.3},
  'Fever': {'Healthy': 0.2, 'Fever': 0.5, 'Dizzy': 0.3},
  'Dizzy': {'Healthy': 0.1, 'Fever': 0.2, 'Dizzy': 0.7}
}
EMISSION_PROB_2_ = {
  'Healthy': {'Normal': 0.6, 'Cold': 0.2, 'Hot': 0.2},
  'Fever': {'Normal': 0.1, 'Cold': 0.7, 'Hot': 0.2},
  'Dizzy': {'Normal': 0.3, 'Cold': 0.3, 'Hot': 0.4}
}

INITIAL_PROB_2_ = {'Healthy': 0.4, 'Fever': 0.3, 'Dizzy': 0.3}
OBSERVATION_SEQ_2_ = ['Normal', 'Cold', 'Hot', 'Normal', 'Cold']
