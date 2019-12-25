import numpy as np

from util import accuracy
from hmm import HMM
from collections import defaultdict

# TODO:
def model_training(train_data, tags):
    """
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
    model = None
	###################################################
	# Edit here
    words_set = set()
    word_state_to_pos_emmision = defaultdict(lambda: defaultdict(int))
    transitions_state_state = defaultdict(lambda: defaultdict(int))
    tag_count = defaultdict(int)
    first_word_count = defaultdict(int)
    transition_count = defaultdict(int)
    
    n_lines = 0
    for sentence in train_data:
        previous_state = ''
        n_lines = n_lines + 1
        first_word_count[sentence.tags[0]] = first_word_count[sentence.tags[0]] + 1
        for i,word in enumerate(sentence.words):
            words_set.add(word)
            tag = sentence.tags[i]
            tag_count[tag] += 1
            word_state_to_pos_emmision[tag][word] += 1
            transitions_state_state[previous_state][tag] += 1
            transition_count[previous_state] += 1
            previous_state = tag

    
    state_dict = dict()
    obs_dict = dict()

    for i, state in enumerate(words_set):
        obs_dict[state] = i

    for i,tag in enumerate(tags):
        state_dict[tag] = i

    pi = [0] * len(state_dict)
    A = [[0] * len(state_dict) for _ in range(len(state_dict))]
    B = [[0] * len(obs_dict) for _ in range(len(state_dict))]
    

    for tag in tags:
        pi[state_dict[tag]] = first_word_count[tag] / len(train_data)

        for next_tag in tags:
            A[state_dict[tag]][state_dict[next_tag]] = transitions_state_state[tag][next_tag] / transition_count[tag]

        for word, count in word_state_to_pos_emmision[tag].items():
            B[state_dict[tag]][obs_dict[word]] = count / tag_count[tag]

    model = HMM(np.array(pi), np.array(A), np.array(B), obs_dict, state_dict)
    
    
#    model = HMM()
    return model

	###################################################

# TODO:
#def sentence_tagging(test_data, model, tags):
#    
#	tagging = []
#    
#    # new_observations = 0
#    len_observations, new_observations = len(model.obs_dict), 0
#    for sentence in test_data:
#      for word in sentence.words:
#          if word not in model.obs_dict:
#              model.obs_dict[word] = len_observations
#              len_observations += 1
#              new_observations += 1
#    
#    matrix_new_observations = np.full((model.B.shape[0], new_observations), (10 ** -6))
#    model.B = np.concatenate((model.B, matrix_new_observations), axis = 1)
#    
#    for sentence in test_data:
#        tagging.append(model.viterbi(sentence.words))
#	###################################################
#	return tagging
    
def sentence_tagging(test_data, model, tags):
    """
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
    tagging = []
    ############################################
    #write here
    len_observations, new_observations = len(model.obs_dict), 0
#    for sentence in test_data:
#    	for word in sentence.words:
#    		if word not in model.obs_dict:
#    			model.obs_dict[word] = len_observations
#                len_observations += 1
#                new_observations += 1
    for sentence in test_data:
        for word in sentence.words:
            if word not in model.obs_dict:
                model.obs_dict[word] = len_observations
                len_observations += 1
                new_observations += 1
#    matrix_new_observations = np.full((model.B.shape[0], new_observations), (10 * * -6))
    matrix_new_observations = np.full((model.B.shape[0], new_observations), (10 ** -6))
    model.B = np.append(model.B, matrix_new_observations, axis = 1)
#    model.B = np.concatenate((model.B, matrix_new_observations), axis = 1)

    for sentence in test_data:
    	tagging.append(model.viterbi(sentence.words))
    ###################################################
    return tagging    
