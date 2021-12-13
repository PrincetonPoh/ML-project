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


def generate_viterbi_values(n, current_label, word_list, words_unique, tags_unique, emission_params, transmission_params):
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
        except ValueError:
            current_max_viterbi_value = -sys.float_info.max

        viterbi_val[(n, current_label)] = current_max_viterbi_value
        return

    # Recursive call to generate viterbi_values for (n-1, tag)
    for tag in tags_unique:
        if (n - 1, tag) not in viterbi_val:
            generate_viterbi_values(
                n - 1,
                tag,
                word_list,
                words_unique,
                tags_unique,
                emission_params,
                transmission_params,
            )

    # Use viterbi values from n-1 to generate current viterbi value
    for tag in tags_unique:
        # Here, we use a try-except block because our emission parameters only contain emissions which appeared in our datasets
        # Thus, any unobserved emission will throw a KeyError, however it's value should be -inf, so we just catch the Error and proceed to the next tag
        # If transmission_params gives 0, math.log will throw a valueError, thus we catch it and skip the current tag since 0 means we should never consider it
        try:
            if word_list[n - 1] in words_unique:
                try:
                    value = viterbi_val[(n - 1, tag)] + math.log(
                        emission_params[(current_label, word_list[n - 1])]
                        * transmission_params[tag][current_label]
                    )
                except KeyError:
                    continue
            else:
                value = viterbi_val[(n - 1, tag)] + math.log(
                    emission_params[(current_label, UNK)]
                    * transmission_params[tag][current_label]
                )
        except ValueError:
            continue

        current_max_viterbi_value = max(current_max_viterbi_value, value)

    viterbi_val[(n, current_label)] = current_max_viterbi_value


# function to kickstart viterbi recursive chain, and add (n+1, STOP) to veterbi_values
def start_viterbi(
    word_list, words_unique, tags_unique, emission_params, transmission_params
):
    global viterbi_val
    max_final_viterbi_value = -sys.float_info.max

    n = len(word_list)

    # Recursive call to generate viterbi_values for (n, tag)
    for tag in tags_unique:
        generate_viterbi_values(
            n,
            tag,
            word_list,
            words_unique,
            tags_unique,
            emission_params,
            transmission_params,
        )

    # Use viterbi values from n to generate viterbi value for (n+1, STOP)
    for tag in tags_unique:
        try:
            value = viterbi_val[(n, tag)] + math.log(
                transmission_params[tag][STOP_TAG]
            )
        except ValueError:
            continue
        max_final_viterbi_value = max(max_final_viterbi_value, value)

    viterbi_val[(n + 1, STOP_TAG)] = max_final_viterbi_value


def generate_predictions_viterbi(word_list, tags_unique, transmission_params):
    global viterbi_val

    n = len(word_list)

    generated_tag_list = ['' for i in range(n)]

    # Compute tag for n
    current_best_tag = 'O'
    current_best_tag_value = -sys.float_info.max

    for tag in tags_unique:
        try:
            value = viterbi_val[(n, tag)] + math.log(
                transmission_params[tag][STOP_TAG]
            )
        except ValueError:
            continue
        if value > current_best_tag_value:
            current_best_tag = tag
            current_best_tag_value = value

    generated_tag_list[n - 1] = current_best_tag

    # Generate predictions starting from n-1 going down to 1
    for i in range(n - 1, 0, -1):
        current_best_tag = 'O'
        current_best_tag_value = -sys.float_info.max

        for tag in tags_unique:
            try:
                value = viterbi_val[(i, tag)] + math.log(
                    transmission_params[tag][generated_tag_list[i]]
                )
            except ValueError:
                continue
            if value > current_best_tag_value:
                current_best_tag = tag
                current_best_tag_value = value

        generated_tag_list[i - 1] = current_best_tag
    return generated_tag_list


def write_p2(predicted_file, words_list, tags_list):
    assert len(words_list) == len(tags_list)

    with open(predicted_file, 'w', encoding='utf8') as f:
        for words, tags in zip(
            words_list, tags_list
        ):  # Unpack all sentences and list of tags
            assert len(words) == len(tags)
            for word, tag in zip(words, tags):  # Unpack all words and tags
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

# Run and output Viterbi for ES
ES_predicted_tags_list = []
for word in ES_test_words:
    viterbi_val = {}
    start_viterbi(
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

# Run and output Viterbi for RU
RU_predicted_tags_list = []
for word in RU_test_words:
    viterbi_val = {}
    start_viterbi(
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

write_p2(
    RU_P2, RU_test_words, RU_predicted_tags_list
)

print('ES dataset results---------')
evaluateScores(ES_DEV_OUT_ORIGINAL, ES_P2)

print('\nRU dataset results---------')
evaluateScores(RU_DEV_OUT_ORIGINAL, RU_P2)



# TODO: Part 3
def get_top_scores_from_dictionary(d, k=5):
    return collections.OrderedDict(sorted(d.items(), reverse=True)[:k])


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

    total_viterbi_scores = get_top_scores_from_dictionary(total_viterbi_scores)

    # Generate predictions starting from n-1 going down to 1
    for i in range(n - 1, 0, -1):
        link = {}
        for tags in total_viterbi_scores.values():

            for tag in tags_unique:
                try:
                    value = viterbi_val[(i, tag)] + math.log(
                        transmission_params[tag][tags[0]]
                    )  # we take the first tag because we are working backwards
                    link[value] = [tag] + tags
                except ValueError:
                    continue

        total_viterbi_scores = get_top_scores_from_dictionary(link)
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

# Run and output Viterbi for ES
ES_predicted_tags_list = []
for word in ES_test_words:
    viterbi_val = {}
    start_viterbi(
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

# Run and output Viterbi for RU
RU_predicted_tags_list = []
for word in RU_test_words:
    viterbi_val = {}
    start_viterbi(
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
    unique_tags, transition_pair_count, tags_with_start_stop
):
    unique_tags = [STARTING_TAG] + unique_tags + [STOP_TAG]
    transition = {}
    for u in unique_tags[:-1]:  # omit STOP
        transition_row = {}
        for v in unique_tags[1:]:  # omit START
            # transition row now starts from 1
            transition_row[v] = 1.0
        transition[u] = transition_row

    # populate transition parameters with counts
    for u, v in transition_pair_count:
        transition[u][v] += transition_pair_count[(u, v)]

    # divide transition_count by count_yi, to get probability
    for u, transition_row in transition.items():
        # have to add length of unique tags
        count_yi = count_label(u, tags_with_start_stop) + len(unique_tags) - 1
        # words in training set
        for v, transition_count in transition_row.items():
            if count_yi == 0:
                transition[u][v] = 0.0
            else:
                transition[u][v] = transition_count / count_yi

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

# Run and output Viterbi for ES
ES_predicted_tags_list = []
for word in ES_test_words:
    viterbi_val = {}
    start_viterbi(
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

# Run and output Viterbi for RU
RU_predicted_tags_list = []
for word in RU_test_words:
    viterbi_val = {}
    start_viterbi(
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

# Run and output Viterbi for ES
ES_predicted_tags_list = []
for word in ES_test_words:
    viterbi_val = {}
    start_viterbi(
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

# Run and output Viterbi for RU
RU_predicted_tags_list = []
for word in RU_test_words:
    viterbi_val = {}
    start_viterbi(
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