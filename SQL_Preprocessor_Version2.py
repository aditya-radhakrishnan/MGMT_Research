import numpy as np
from Gensim_Maker import GensimMaker
import random
import sqlite3
import re

class SQLPreprocessorVersion2(object):

    # database location where data is gotten
    db_name = ''

    # name of the table where data is kept
    table_name = ''

    # the name of the column where the data is stored
    column_name = ''

    # location of gensim model that already has embedding matrix or location to save a new embedding matrix
    model_location = ''

    # the delimiter used
    delimiter = ''

    # gensim_model to be used
    gensim_model = None

    # Gensim_Maker object to be used
    gensim_maker_obj = None

    # the maximum number of words any one patent's text is
    max_num_words = 0

    # dimension of any word vector in the model
    word_vector_dimension = 0

    def __init__(self, db_name, table_name, model_location, column_name, gensim_maker_obj=None, delimiter = '',
                 gensim_model=None, need_to_create_models=True):
        self.db_name = db_name
        self.model_location = model_location
        # To protect against injection nonsense
        self.table_name = self.remove_non_alphachars(table_name)
        self.column_name = self.remove_non_alphachars(column_name)
        self.delimiter = delimiter
        if gensim_maker_obj is None:
            self.gensim_maker_obj = GensimMaker(db_name, sql_table_name=table_name, use_SQL_sentence_maker=True)
        else:
            self.gensim_maker_obj = gensim_maker_obj
        if need_to_create_models:
            self.gensim_maker_obj.generate_model()
            self.gensim_maker_obj.save_model(model_location)
        else:
            self.gensim_maker_obj.load_model(model_location)
        if gensim_model is None:
            self.gensim_model = self.gensim_maker_obj.get_model()

        self.word_vector_dimension = self.gensim_maker_obj.get_dimension_of_a_word()

        self.max_num_words = self.find_max_num_words()

    # gets the maximum number of words in any file in the directory and make a list of company names
    def find_max_num_words(self):
        max_length = float('inf') * -1
        conn = sqlite3.connect(self.db_name)
        all_patents = conn.execute("SELECT " + self.column_name + " FROM " + self.table_name + ";")
        for line in all_patents:
            if self.delimiter in str(line[0]):
                curr_length = len(str(line[0]).split(self.delimiter))
            else:
                curr_length = len(str(line[0]).split())
            if curr_length > max_length:
                max_length = curr_length
        conn.close()
        return max_length

    def remove_non_alphachars(self, string_to_clean):
            return ''.join(char for char in string_to_clean if char.isalnum()or char is '_')

    # Converts the patent to an array and pad it to the maximum length text string of any patent's data
    def convert_patent_to_array(self, patent_id):
        num_dim_of_a_word = self.gensim_maker_obj.get_dimension_of_a_word()
        conn = sqlite3.connect(self.db_name)
        query = conn.execute("SELECT " + self.column_name + " FROM " + self.table_name + " WHERE patent_id =\"" + patent_id + "\"")
        line = ''
        for row in query:
            line = str(row[0])
        big_array = np.zeros((num_dim_of_a_word, self.max_num_words))
        word_index = 0
        if self.delimiter in line:
            for word in line.split(self.delimiter):
                try:
                    new_word = self.gensim_model[word]
                # if a word isn't in the model then just leave it as a 0 vector
                except KeyError:
                    word_index += 1
                    continue
                big_array[:, word_index] = new_word
                word_index += 1
            big_array.view('float32')
            conn.close()
            return big_array
        else:
            for word in line.replace('.', '').split():
                try:
                    new_word = self.gensim_model[word]
                # if a word isn't in the model then just leave it as a 0 vector
                except KeyError:
                    word_index += 1
                    continue
                big_array[:, word_index] = new_word
                word_index += 1
            big_array.view('float32')
            conn.close()
            return big_array

    # creates the input matrix by making an array out of the text at that row in the generated db and stacks them on top of each other
    def create_input_matrix(self, list_of_patent_ids):
        matrix_to_return = np.empty((len(list_of_patent_ids), self.gensim_maker_obj.get_dimension_of_a_word(), self.max_num_words, 1))
        matrix_depth_index = 0
        for patent in list_of_patent_ids:
            matrix_to_add = self.convert_patent_to_array(patent)
            matrix_to_return[matrix_depth_index, :, :, 0] = matrix_to_add
            matrix_depth_index += 1
        matrix_to_return.view('float32')
        return matrix_to_return

    # returns a list of training set, validation set, and test set as 4d numpy arrays as a list in the following order:
    # training input, validation input, test input.
    # input is a list of 3 numbers which are the ratios of the training, validation, and test set, in order
    def get_input_set(self, ratios_list):
        random_patents_lists = self.get_random_patent_ids(ratios_list)
        return self.get_input_set_given_patent_ids(random_patents_lists)

    def get_input_set_given_patent_ids(self, list_of_patent_ids):

        train_set_patent_list = list_of_patent_ids[0]
        valid_set_patent_list = list_of_patent_ids[1]
        test_set_patent_list = list_of_patent_ids[2]

        return [self.create_input_matrix(train_set_patent_list),
                self.create_input_matrix(valid_set_patent_list),
                self.create_input_matrix(test_set_patent_list)]

    # given ratios for training, validation, and test data, returns 3 lists each containing random patent ids
    def get_random_patent_ids(self, ratios_list):
        conn = sqlite3.connect(self.db_name)
        query = conn.execute("SELECT * FROM " + self.table_name + ";")
        patent_id_list = list()
        for row in query:
            patent_id_list.append(str(row[0]))

        num_patents_in_file = len(patent_id_list)

        num_patents_for_train_set = int(ratios_list[0] * num_patents_in_file)
        num_patents_for_valid_set = int(ratios_list[1] * num_patents_in_file)
        if (ratios_list[0] + ratios_list[1] + ratios_list[2] == 1):
            num_patents_for_test_set = num_patents_in_file - num_patents_for_train_set - num_patents_for_valid_set
        else:
            num_patents_for_test_set = int(ratios_list[2] * num_patents_in_file)

        train_set_patents_list = list()
        valid_set_patents_list = list()
        test_set_patents_list = list()

        for count in range(num_patents_for_train_set):
            random_patent = random.choice(patent_id_list)
            train_set_patents_list.append(random_patent)
            patent_id_list.remove(random_patent)

        for count in range(num_patents_for_valid_set):
            random_patent = random.choice(patent_id_list)
            valid_set_patents_list.append(random_patent)
            patent_id_list.remove(random_patent)

        for count in range(num_patents_for_test_set):
            random_patent = random.choice(patent_id_list)
            test_set_patents_list.append(random_patent)
            patent_id_list.remove(random_patent)

        conn.close()

        return [train_set_patents_list, valid_set_patents_list, test_set_patents_list]

'''
preprocessor_test = SQLPreprocessor('test.db',
                                          'patents',
                                          '/home/trehuang/Desktop/preliminary_lists/crap_test_model2.txt',
                                          'organized_patents',
                                    gensim_maker_obj= GensimMaker('test.db', sql_table_name='patents', min_to_ignore=1, size=20, use_SQL_sentence_maker=True),
                                    table_already_exists=True,
                                    need_to_create_models= True)
# print(preprocessor_test.answer_list)
# print preprocessor_test.one_hot_dict
print(preprocessor_test.get_patent_ids_from_line_numbers([0, 2, 3]))
'''