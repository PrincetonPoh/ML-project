STARTING_TAG = 'START'
STOP_TAG = 'STOP'


def process_train_file_part_1(file_path):
    with open(file_path, encoding="utf8") as f:
        data = f.read().splitlines()
        data[:] = [x for x in data if x]

    output = []
    for i in data:
        i = i.split(" ")
        if len(i) > 2:
            i = [" ".join(i[0 : len(i) - 1]), i[len(i) - 1]]
            output.append(i)
        else:
            output.append(i)
    # f.close()
    return output


def process_test_file_part_1(path):
    with open(path, encoding="utf8") as f:
        data = f.read().splitlines()

    output = []
    for word in data:
        output.append(word)

    # f.close()
    return output


def process_train_file_part_2(training_file):
    labels = []
    labels_with_start_stop = []
    words = []

    with open(training_file, "r", encoding="utf8") as f:
        doc = f.read().rstrip()
        lines = doc.split("\n\n")

        for line in lines:
            labels_list = []
            labels_with_start_stop_list = []
            words_list = []

            for word_tag in line.split("\n"):
                i = word_tag.split(" ")

                if len(i) > 2:
                    i = [" ".join(i[0 : len(i) - 1]), i[len(i) - 1]]

                word, tag = i[0], i[1]
                words_list.append(word)
                labels_list.append(tag)

            labels.append(labels_list)
            labels_with_start_stop_list = [STARTING_TAG] + labels_list + [STOP_TAG]
            labels_with_start_stop.append(labels_with_start_stop_list)
            words.append(words_list)

    # f.close()
    return labels, labels_with_start_stop, words


def process_test_file_part_2(testing_file):
    test_words = []

    with open(testing_file, encoding="utf8") as f:
        doc = f.read().rstrip()
        lines = doc.split("\n\n")

        for line in lines:
            word_list = []
            for word in line.split("\n"):
                word_list.append(word)
            test_words.append(word_list)

    # f.close()
    return test_words
