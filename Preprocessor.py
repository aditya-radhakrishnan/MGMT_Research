import numpy as np
from Gensim_Maker import GensimMaker
import random
import re

class Preprocessor(object):

    # file location
    file_location = ''

    # the file lines
    lines = None

    # Location of gensim model that already has embedding matrix or location to save a new embedding matrix
    model_location = ''

    # gensim_model to be used
    gensim_model = None

    # Gensim_Maker object to be used
    gensim_maker_obj = None

    # the maximum number of words any one company's text is
    max_num_words = 0

    # the dictionary of a word and its one-hot encoding
    one_hot_dict = dict()

    def __init__(self, file_location, model_location, gensim_model = None, need_to_create_model = True):
        self.file_location = file_location
        self.model_location = model_location
        self.gensim_maker_obj = GensimMaker(file_location)
        if (need_to_create_model):
            self.gensim_maker_obj.generate_model()
            self.gensim_maker_obj.save_model(model_location)
        else:
            self.gensim_maker_obj.load_model(model_location)
        if gensim_model is None:
            self.gensim_model = self.gensim_maker_obj.get_model()

        with open(self.file_location) as file:
            self.lines = file.readlines()

        company_name_set = set()

        # get the maximum number of words in any file in the directory and make a list of company names
        max_length = float('inf') * -1
        for line in self.lines:
            line_split_list = line.split('\t')
            company_name = line_split_list[0]
            text_body = line_split_list[1]
            length = len(text_body.split())
            if length > max_length:
                max_length = length
            self.max_num_words = max_length
            company_name_set.add(company_name)
        self.one_hot_dict = self.create_one_hot_encoding_dictionary(company_name_set)

    # Converts line of a file to an array and pad it to a certain length
    def convert_line_of_file_to_array(self, line_number):
        num_dim_of_a_word = self.gensim_maker_obj.get_dimension_of_a_word()
        line = self.lines[line_number]
        text_body = line.split('\t')[1]
        big_array = np.zeros((num_dim_of_a_word, self.max_num_words))
        word_index = 0
        for word in text_body.split():
            try:
                new_word = self.gensim_model[word]
            # if a word isn't in the model then just leave it as a 0 vector
            except KeyError:
                word_index += 1
                continue
            big_array[:, word_index] = new_word
            word_index += 1
        big_array.view('float32')
        return big_array

    # creates the input matrix by making an array out of each file and stacking them on top of each other
    def create_input_matrix(self, list_of_line_numbers):
        matrix_to_return = np.empty((len(list_of_line_numbers), self.gensim_maker_obj.get_dimension_of_a_word(), self.max_num_words, 1))
        matrix_depth_index = 0
        for line_number in list_of_line_numbers:
            matrix_to_add = self.convert_line_of_file_to_array(line_number)
            matrix_to_return[matrix_depth_index, :, :, 0] = matrix_to_add
            matrix_depth_index += 1
        matrix_to_return.view('float32')
        return matrix_to_return

    # creates the correct answer matrix by vertically stacking the word vectors of the input list of answers
    def create_answer_matrix(self, list_of_line_numbers):
        matrix_to_return = np.empty((len(list_of_line_numbers), len(self.one_hot_dict)))
        matrix_row_index = 0
        for line_number in list_of_line_numbers:
            line = self.lines[line_number]
            company_name = line.split('\t')[0]
            matrix_to_return[matrix_row_index, :] = self.one_hot_dict[company_name]
            matrix_row_index += 1
        matrix_to_return.view('float32')
        return matrix_to_return

    # creates the dictionary for the one hot vectors
    def create_one_hot_encoding_dictionary(self, list_of_words):
        vector_length = len(list_of_words)
        dictionary = dict()
        index = 0
        for name in list_of_words:
            vector = np.zeros((1, vector_length))
            vector.view('float32')
            vector[0, index] = 1
            dictionary[name] = vector
            index += 1
        return dictionary

    # returns a list of training set, validation set, and test set as 4d numpy arrays as a list in the following order:
    # training input, training labels, validation input, validation labels, test input, test labels.
    # input is a list of 3 numbers which are the ratios of the training, validation, and test set, in order
    def get_data_set(self, ratios_list):
        num_lines_in_file = len(self.lines)
        num_lines_for_train_set = int(ratios_list[0] * num_lines_in_file)
        num_lines_for_valid_set = int(ratios_list[1] * num_lines_in_file)
        num_lines_for_test_set = num_lines_in_file - num_lines_for_train_set - num_lines_for_valid_set

        train_set_lines_list = list()
        valid_set_lines_list = list()
        test_set_lines_list = list()

        lines_list = list(range(num_lines_in_file))
        for count in range(num_lines_for_train_set):
            random_line = random.choice(lines_list)
            train_set_lines_list.append(random_line)
            lines_list.remove(random_line)

        for count in range(num_lines_for_valid_set):
            random_line = random.choice(lines_list)
            valid_set_lines_list.append(random_line)
            lines_list.remove(random_line)

        for count in range(num_lines_for_test_set):
            random_line = random.choice(lines_list)
            test_set_lines_list.append(random_line)
            lines_list.remove(random_line)

        return [self.create_input_matrix(train_set_lines_list), self.create_answer_matrix(train_set_lines_list),
                self.create_input_matrix(valid_set_lines_list), self.create_answer_matrix(valid_set_lines_list),
                self.create_input_matrix(test_set_lines_list), self.create_answer_matrix(test_set_lines_list)]

    def get_data_set_given_line_numbers(self, list_of_line_numbers):

        train_set_lines_list = list_of_line_numbers[0]
        valid_set_lines_list = list_of_line_numbers[1]
        test_set_lines_list = list_of_line_numbers[2]

        return [self.create_input_matrix(train_set_lines_list), self.create_answer_matrix(train_set_lines_list),
                self.create_input_matrix(valid_set_lines_list), self.create_answer_matrix(valid_set_lines_list),
                self.create_input_matrix(test_set_lines_list), self.create_answer_matrix(test_set_lines_list)]

    def get_random_line_numbers(self, ratios_list):
        num_lines_in_file = len(self.lines)
        num_lines_for_train_set = int(ratios_list[0] * num_lines_in_file)
        num_lines_for_valid_set = int(ratios_list[1] * num_lines_in_file)
        num_lines_for_test_set = num_lines_in_file - num_lines_for_train_set - num_lines_for_valid_set

        train_set_lines_list = list()
        valid_set_lines_list = list()
        test_set_lines_list = list()

        lines_list = list(range(num_lines_in_file))
        for count in range(num_lines_for_train_set):
            random_line = random.choice(lines_list)
            train_set_lines_list.append(random_line)
            lines_list.remove(random_line)

        for count in range(num_lines_for_valid_set):
            random_line = random.choice(lines_list)
            valid_set_lines_list.append(random_line)
            lines_list.remove(random_line)

        for count in range(num_lines_for_test_set):
            random_line = random.choice(lines_list)
            test_set_lines_list.append(random_line)
            lines_list.remove(random_line)

        return [train_set_lines_list, valid_set_lines_list, test_set_lines_list]

    # removes the numbers and extension '.txt' from a string
    def remove_numbers_and_extension(self, word):
        output = word.replace('.txt', '')
        output = re.sub('([0-9]*)', '', output)
        return output

# preprocessor_test = Preprocessing('/home/trehuang/Desktop/Earnings Calls Data/Split_Documents/part1.txt', '/home/trehuang/Desktop/Earnings Calls Data/Split_Documents/part1_model.txt', need_to_create_model = False)
# print(preprocessor_test.create_answer_matrix(('aapl', 'abb', 'aapl')))
# print(preprocessor_test.answer_list)
# print preprocessor_test.one_hot_dict
# print(preprocessor_test.get_data_set([.01, .6, .39])[0].shape)
