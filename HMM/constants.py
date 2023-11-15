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