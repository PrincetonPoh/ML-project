from evaluate import *
from file_processing import *
import math
import sys
import itertools
import collections

ES_TRAIN = './ES/train'
ES_TEST = './ES/dev.in'
ES_DEV_OUT_ORIGINAL = './ES/dev.out'
ES_P1 = './ES/dev.p1.out'
ES_P2 = './ES/dev.p2.out'
ES_P3 = './ES/dev.p3.out'
ES_P4 = './ES/dev.p4.out'
ES_TEST_IN = './ES/test.in'
ES_TEST_OUT = './ES/test.out'

RU_TRAIN = './RU/train'
RU_TEST = './RU/dev.in'
RU_DEV_OUT_ORIGINAL = './RU/dev.out'
RU_P1 = './RU/dev.p1.out'
RU_P2 = './RU/dev.p2.out'
RU_P3 = './RU/dev.p3.out'
RU_P4 = './RU/dev.p4.out'
RU_TEST_IN = './RU/test.in'
RU_TEST_OUT = './RU/test.out'

UNK = '#UNK#'
STARTING_TAG = 'START'
STOP_TAG = 'STOP'

# TODO: Part 1
def mle_to_emission(training_dict):
    labels = {}
    labels_words = {}
    emission_params = {}

    for i in training_dict:
        word, label = i[0], i[1]

        # get count(y)
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1

        # get count (y->x) 
        label_to_word = (label, word)
        if label_to_word in labels_words:
            labels_words[label_to_word] += 1
        else:
            labels_words[label_to_word] = 1

    for key in labels_words.keys():
        emission_params[key] = labels_words[key] / (labels[key[0]] + 1)
    for key in labels.keys():

        # handle situation where the word dont exist using '#UNK#'
        transition = (key, UNK)
        emission_params[transition] = 1 / (labels[key] + 1)

    return emission_params


def simple_sentiment_analysis(emission_params):
    tempt_most_probable = {}
    answer = {}
    # interate through every word and label
    for i, prob in emission_params.items():
        label, word = i[0], i[1]
        
        if (word) not in (tempt_most_probable):
            tempt_most_probable[word] =prob
            answer[word] =  label
        else:
            if prob > tempt_most_probable[word]:
                tempt_most_probable[word] = prob
                answer[word] = label

    return answer


def write_p1(predicted_file, test_file, highest_prob_tag):
    f = open(predicted_file, 'w', encoding='utf8')
    for word in test_file:
        if len(word) > 0:
            try:
                label = highest_prob_tag[word]
            except:
                # handle missing word
                label = highest_prob_tag[UNK]
            f.write(f'{word} {label}\n')
        else:
            # next line
            f.write('\n')
    f.close()


print('Part 1 ###################################################')
# ES dataset
ES_train_data = process_train_file_part_1(ES_TRAIN)
ES_test_data = process_test_file_part_1(ES_TEST)
ES_emission_parameters = mle_to_emission(ES_train_data)
ES_most_probable_tag = simple_sentiment_analysis(ES_emission_parameters)
write_p1(ES_P1, ES_test_data, ES_most_probable_tag)

# RU dataset
RU_train_data = process_train_file_part_1(RU_TRAIN)
RU_test_data = process_test_file_part_1(RU_TEST)
RU_emission_parameters = mle_to_emission(RU_train_data)
RU_most_probable_tag = simple_sentiment_analysis(RU_emission_parameters)
write_p1(RU_P1, RU_test_data, RU_most_probable_tag)

print('ES dataset results---------')
evaluateScores(ES_DEV_OUT_ORIGINAL, ES_P1)
print('\nRU dataset results---------')
evaluateScores(RU_DEV_OUT_ORIGINAL, RU_P1)



# TODO: Part 2
def unique_element(x):
    # need to do it this way to access the elements inside of the tuple key
    temp = list(itertools.chain.from_iterable(x))
    return list(set(temp))

def transition_pairs(labels):
    count = {}

    for label in labels:
        # current and previous label
        for label_one, label_two in zip(label[:-1], label[1:]):
            transition_pair = (label_one, label_two)
            if transition_pair in count:
                count[transition_pair] += 1
            else:
                count[transition_pair] = 1

    return count


def count_label(label, labels):
    labels_flattened = list(itertools.chain.from_iterable(labels))
    return labels_flattened.count(label)


def mle_to_transition(unique_tags, t_pair_count, labels_with_start_stop):
    unique_tags = [STARTING_TAG] + unique_tags + [STOP_TAG]
    transition_dict = {}
    
    # remove STOP in sequence
    for i in unique_tags[:-1]:
    # for u in unique_tags[:]: 
    
        transition_row = {}

        # remove START in sequence
        for j in unique_tags[1:]:
        # for v in unique_tags[:]:

            transition_row[j] = 0.0

        transition_dict[i] = transition_row

    # add counts to transition dictionary
    for i, j in t_pair_count:
        transition_dict[i][j] += t_pair_count[(i, j)]

    # Get probability
    for i, transition_row in transition_dict.items():
        count_label_i = count_label(i, labels_with_start_stop)

        # words in training set
        for j, transition_count in transition_row.items():

            if count_label_i == 0:
                transition_dict[i][j] = 0.0
            else:
                # transition count divided by count_label
                transition_dict[i][j] = transition_count / count_label_i

    return transition_dict


list_out_word_output = []  # list of (word, predicted label)
viterbi_val = {}  # {(n, label): float}


def generate_viterbi_values(n, current_label, word_list, words_unique, labels_unique, emission_params, transmission_params):
    global viterbi_val

    # Smallest possible float
    current_max_viterbi_value = -sys.float_info.max

    if n == 0:
        return
    elif n == 1:
        try:
            if word_list[n - 1] in words_unique:
                try:
                    current_max_viterbi_value = math.log(
                        emission_params[(current_label, word_list[n - 1])] * transmission_params[STARTING_TAG][current_label]
                    )
                except KeyError:
                    current_max_viterbi_value = -sys.float_info.max
            else:
                current_max_viterbi_value = math.log(
                    emission_params[(current_label, UNK)]
                    * transmission_params[STARTING_TAG][current_label]
                )
        # handles the value error with wrong values 
        except ValueError:
            current_max_viterbi_value = -sys.float_info.max

        viterbi_val[(n, current_label)] = current_max_viterbi_value
        return

    for label in labels_unique:
        # create viterbi values (n-1, label)
        if (n - 1, label) not in (viterbi_val):
            generate_viterbi_values( (n - 1), label, word_list, words_unique, labels_unique, emission_params, transmission_params)

    for label in labels_unique:
        try:
            if word_list[n - 1] in words_unique:
                try:
                    value = viterbi_val[(n - 1, label)] + math.log(
                        emission_params[(current_label, word_list[n - 1])] * transmission_params[label][current_label])
                # handle case where it isnt inside the dataset
                except KeyError:
                    continue
            else:
                value = viterbi_val[(n - 1, label)] + math.log( emission_params[(current_label, UNK)] * transmission_params[label][current_label])
        # handle zero error
        except ValueError:
            continue

        current_max_viterbi_value = max(current_max_viterbi_value, value)

    viterbi_val[(n, current_label)] = current_max_viterbi_value


def start_v(word_list, words_unique, labels_unique, emission_params, transmission_params):
    global viterbi_val

    # Smallest possible float
    max_final_v_value = -sys.float_info.max

    n = len(word_list)

    for label in labels_unique:
        generate_viterbi_values(n,label,word_list,words_unique,labels_unique,emission_params,transmission_params)

    # viterbi values for (n+1, stop)
    for label in labels_unique:
        try:
            value = viterbi_val[(n, label)] + math.log(transmission_params[label][STOP_TAG])
        except ValueError:
            continue
        max_final_v_value = max(max_final_v_value, value)

    viterbi_val[(n + 1, STOP_TAG)] = max_final_v_value


def generate_predictions_viterbi(word_list, labels_unique, trans_params):
    global viterbi_val

    n = len(word_list)

    generated_label_list = ['' for i in range(n)]
    current_best_label = 'O'
    current_best_label_value = -sys.float_info.max

    for label in labels_unique:
        try:
            value = viterbi_val[(n, label)] + math.log(trans_params[label][STOP_TAG])
        except ValueError:
            continue
        # to get max value
        if value > current_best_label_value:
            current_best_label = label
            current_best_label_value = value

    generated_label_list[n - 1] = current_best_label


    # get predictions from the back
    for i in range(n - 1, 0, -1):
        current_best_label = 'O'
        current_best_label_value = -sys.float_info.max

        for label in labels_unique:
            try:
                value = viterbi_val[(i, label)] + math.log(
                    trans_params[label][generated_label_list[i]]
                )
            except ValueError:
                continue
            if value > current_best_label_value:
                current_best_label = label
                current_best_label_value = value

        generated_label_list[i - 1] = current_best_label

    return generated_label_list


def write_p2(predicted_file, words_list, tags_list):
    assert len(words_list) == len(tags_list)

    with open(predicted_file, 'w', encoding='utf8') as f:
        # unpack
        for words, tags in zip(words_list, tags_list): 
            assert len(words) == len(tags)
            for word, tag in zip(words, tags):
                f.write(f'{word} {tag}\n')
            f.write('\n')


print('Part 2 ###################################################')
# ES dataset
ES_train_data = process_train_file_part_1(ES_TRAIN)
ES_emission_parameters = mle_to_emission(ES_train_data)
ES_tags, ES_tags_with_start_stop, ES_train_words = process_train_file_part_2(ES_TRAIN)
ES_test_words = process_test_file_part_2(ES_TEST)
ES_unique_words = unique_element(ES_train_words)
ES_unique_tags = unique_element(ES_tags)
ES_transition_pair_count = transition_pairs(ES_tags_with_start_stop)
ES_transition_parameters = mle_to_transition(ES_unique_tags, ES_transition_pair_count, ES_tags_with_start_stop)

# Viterbi for ES
ES_predicted_tags_list = []
for word in ES_test_words:
    viterbi_val = {}
    start_v(
        word,
        ES_unique_words,
        ES_unique_tags,
        ES_emission_parameters,
        ES_transition_parameters,
    )
    ES_generated_tag_list = generate_predictions_viterbi(
        word, ES_unique_tags, ES_transition_parameters
    )
    ES_predicted_tags_list.append(ES_generated_tag_list)

write_p2(
    ES_P2, ES_test_words, ES_predicted_tags_list
)

# RU dataset
RU_train_data = process_train_file_part_1(RU_TRAIN)
RU_emission_parameters = mle_to_emission(RU_train_data)
RU_tags, RU_tags_with_start_stop, RU_train_words = process_train_file_part_2(RU_TRAIN)
RU_test_words = process_test_file_part_2(RU_TEST)
RU_unique_words = unique_element(RU_train_words)
RU_unique_tags = unique_element(RU_tags)
RU_transition_pair_count = transition_pairs(RU_tags_with_start_stop)
RU_transition_parameters = mle_to_transition(RU_unique_tags, RU_transition_pair_count, RU_tags_with_start_stop)

# Viterbi for RU
RU_predicted_tags_list = []
for word in RU_test_words:
    viterbi_val = {}
    start_v(
        word,
        RU_unique_words,
        RU_unique_tags,
        RU_emission_parameters,
        RU_transition_parameters,
    )
    RU_generated_tag_list = generate_predictions_viterbi(
        word, RU_unique_tags, RU_transition_parameters
    )
    RU_predicted_tags_list.append(RU_generated_tag_list)

write_p2(RU_P2, RU_test_words, RU_predicted_tags_list)

print('ES dataset results---------')
evaluateScores(ES_DEV_OUT_ORIGINAL, ES_P2)
print('\nRU dataset results---------')
evaluateScores(RU_DEV_OUT_ORIGINAL, RU_P2)



# TODO: Part 3
def get_top_5(d):
    # get top 5
    return collections.OrderedDict(sorted(d.items(), reverse=True)[:5])


def generate_predictions_viterbi_part_3(word_list, tags_unique, transmission_params):
    global viterbi_val
    total_viterbi_scores = {}
    n = len(word_list)

    for tag in tags_unique:
        try:
            value = viterbi_val[(n, tag)] + math.log(
                transmission_params[tag][STOP_TAG]
            )
            total_viterbi_scores[value] = [tag]
        except ValueError:
            continue

    total_viterbi_scores = get_top_5(total_viterbi_scores)

    for i in range(n - 1, 0, -1):
        link = {}
        for tags in total_viterbi_scores.values():

            for tag in tags_unique:
                try:
                    value = viterbi_val[(i, tag)] + math.log(transmission_params[tag][tags[0]])
                    link[value] = [tag] + tags
                except ValueError:
                    continue

        total_viterbi_scores = get_top_5(link)
    return list(total_viterbi_scores.values())[-1]


print('Part 3 ###################################################')
ES_train_data = process_train_file_part_1(ES_TRAIN)
ES_emission_parameters = mle_to_emission(ES_train_data)

ES_tags, ES_tags_with_start_stop, ES_train_words = process_train_file_part_2(
    ES_TRAIN
)
ES_test_words = process_test_file_part_2(ES_TEST)
ES_unique_words = unique_element(ES_train_words)
ES_unique_tags = unique_element(ES_tags)

ES_transition_pair_count = transition_pairs(ES_tags_with_start_stop)
ES_transition_parameters = mle_to_transition(
    ES_unique_tags, ES_transition_pair_count, ES_tags_with_start_stop
)

RU_train_data = process_train_file_part_1(RU_TRAIN)
RU_emission_parameters = mle_to_emission(RU_train_data)

RU_tags, RU_tags_with_start_stop, RU_train_words = process_train_file_part_2(
    RU_TRAIN
)
RU_test_words = process_test_file_part_2(RU_TEST)
RU_unique_words = unique_element(RU_train_words)
RU_unique_tags = unique_element(RU_tags)

RU_transition_pair_count = transition_pairs(RU_tags_with_start_stop)
RU_transition_parameters = mle_to_transition(
    RU_unique_tags, RU_transition_pair_count, RU_tags_with_start_stop
)

# Viterbi for ES
ES_predicted_tags_list = []
for word in ES_test_words:
    viterbi_val = {}
    start_v(
        word,
        ES_unique_words,
        ES_unique_tags,
        ES_emission_parameters,
        ES_transition_parameters,
    )
    ES_generated_tag_list = generate_predictions_viterbi_part_3(
        word, ES_unique_tags, ES_transition_parameters
    )
    ES_predicted_tags_list.append(ES_generated_tag_list)

write_to_output_file(
    ES_P3, ES_test_words, ES_predicted_tags_list
)

# Viterbi for RU
RU_predicted_tags_list = []
for word in RU_test_words:
    viterbi_val = {}
    start_v(
        word,
        RU_unique_words,
        RU_unique_tags,
        RU_emission_parameters,
        RU_transition_parameters,
    )
    RU_generated_tag_list = generate_predictions_viterbi_part_3(
        word, RU_unique_tags, RU_transition_parameters
    )
    RU_predicted_tags_list.append(RU_generated_tag_list)

write_to_output_file(
    RU_P3, RU_test_words, RU_predicted_tags_list
)

print('ES dataset results---------')
evaluateScores(ES_DEV_OUT_ORIGINAL, ES_P3)

print('\nRU dataset results---------')
evaluateScores(RU_DEV_OUT_ORIGINAL, RU_P3)



# TODO: Part 4
def get_transition_using_add1_estimate(
    unique_labels, transition_pair_count, labels_with_start_stop
):
    unique_labels = [STARTING_TAG] + unique_labels + [STOP_TAG]
    transition = {}
    for i in unique_labels[:-1]:
        transition_row = {}
        for j in unique_labels[1:]:
            transition_row[j] = 1.0
        transition[i] = transition_row

    for i, j in transition_pair_count:
        transition[i][j] += transition_pair_count[(i, j)]

    for i, transition_row in transition.items():
        count_yi = count_label(i, labels_with_start_stop) + len(unique_labels) - 1
        for j, transition_count in transition_row.items():
            if count_yi == 0:
                transition[i][j] = 0.0
            else:
                transition[i][j] = transition_count / count_yi

    return transition


print('Part 4 (i) ###################################################')
ES_train_data = process_train_file_part_1(ES_TRAIN)
ES_emission_parameters = mle_to_emission(ES_train_data)

ES_tags, ES_tags_with_start_stop, ES_train_words = process_train_file_part_2(
    ES_TRAIN
)
ES_test_words = process_test_file_part_2(ES_P4)
ES_unique_words = unique_element(ES_train_words)
ES_unique_tags = unique_element(ES_tags)

ES_transition_pair_count = transition_pairs(ES_tags_with_start_stop)
ES_transition_parameters = get_transition_using_add1_estimate(
    ES_unique_tags, ES_transition_pair_count, ES_tags_with_start_stop
)

RU_train_data = process_train_file_part_1(RU_TRAIN)
RU_emission_parameters = mle_to_emission(RU_train_data)

RU_tags, RU_tags_with_start_stop, RU_train_words = process_train_file_part_2(
    RU_TRAIN
)
RU_test_words = process_test_file_part_2(RU_P4)
RU_unique_words = unique_element(RU_train_words)
RU_unique_tags = unique_element(RU_tags)

RU_transition_pair_count = transition_pairs(RU_tags_with_start_stop)
RU_transition_parameters = get_transition_using_add1_estimate(
    RU_unique_tags, RU_transition_pair_count, RU_tags_with_start_stop
)

# Viterbi for ES
ES_predicted_tags_list = []
for word in ES_test_words:
    viterbi_val = {}
    start_v(
        word,
        ES_unique_words,
        ES_unique_tags,
        ES_emission_parameters,
        ES_transition_parameters,
    )
    ES_generated_tag_list = generate_predictions_viterbi(
        word, ES_unique_tags, ES_transition_parameters
    )
    ES_predicted_tags_list.append(ES_generated_tag_list)

write_to_output_file(
    ES_P4, ES_test_words, ES_predicted_tags_list
)

# Viterbi for RU
RU_predicted_tags_list = []
for word in RU_test_words:
    viterbi_val = {}
    start_v(
        word,
        RU_unique_words,
        RU_unique_tags,
        RU_emission_parameters,
        RU_transition_parameters,
    )
    RU_generated_tag_list = generate_predictions_viterbi(
        word, RU_unique_tags, RU_transition_parameters
    )
    RU_predicted_tags_list.append(RU_generated_tag_list)

write_to_output_file(
    RU_P4, RU_test_words, RU_predicted_tags_list
)

print('ES dataset results---------')
evaluateScores(ES_DEV_OUT_ORIGINAL, ES_P4)

print('\nRU dataset results---------')
evaluateScores(RU_DEV_OUT_ORIGINAL, RU_P4)


print('Part 4 (ii) ###################################################')
ES_train_data = process_train_file_part_1(ES_TRAIN)
ES_emission_parameters = mle_to_emission(ES_train_data)

ES_tags, ES_tags_with_start_stop, ES_train_words = process_train_file_part_2(
    ES_TRAIN
)
ES_test_words = process_test_file_part_2(ES_TEST_IN)
ES_unique_words = unique_element(ES_train_words)
ES_unique_tags = unique_element(ES_tags)

ES_transition_pair_count = transition_pairs(ES_tags_with_start_stop)
ES_transition_parameters = get_transition_using_add1_estimate(
    ES_unique_tags, ES_transition_pair_count, ES_tags_with_start_stop
)

RU_train_data = process_train_file_part_1(RU_TRAIN)
RU_emission_parameters = mle_to_emission(RU_train_data)

RU_tags, RU_tags_with_start_stop, RU_train_words = process_train_file_part_2(
    RU_TRAIN
)
RU_test_words = process_test_file_part_2(RU_P4)
RU_unique_words = unique_element(RU_train_words)
RU_unique_tags = unique_element(RU_tags)

RU_transition_pair_count = transition_pairs(RU_tags_with_start_stop)
RU_transition_parameters = get_transition_using_add1_estimate(
    RU_unique_tags, RU_transition_pair_count, RU_tags_with_start_stop
)

# Viterbi for ES
ES_predicted_tags_list = []
for word in ES_test_words:
    viterbi_val = {}
    start_v(
        word,
        ES_unique_words,
        ES_unique_tags,
        ES_emission_parameters,
        ES_transition_parameters,
    )
    ES_generated_tag_list = generate_predictions_viterbi(
        word, ES_unique_tags, ES_transition_parameters
    )
    ES_predicted_tags_list.append(ES_generated_tag_list)

write_to_output_file(
    ES_TEST_OUT, ES_test_words, ES_predicted_tags_list
)

# Viterbi for RU
RU_predicted_tags_list = []
for word in RU_test_words:
    viterbi_val = {}
    start_v(
        word,
        RU_unique_words,
        RU_unique_tags,
        RU_emission_parameters,
        RU_transition_parameters,
    )
    RU_generated_tag_list = generate_predictions_viterbi(
        word, RU_unique_tags, RU_transition_parameters
    )
    RU_predicted_tags_list.append(RU_generated_tag_list)

write_to_output_file(
    RU_TEST_OUT, RU_test_words, RU_predicted_tags_list
)