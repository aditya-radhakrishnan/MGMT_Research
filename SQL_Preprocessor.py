import numpy as np
from Gensim_Maker import GensimMaker
import random
import sqlite3
import re

class SQLPreprocessor(object):

    # database location where data is gotten
    db_name = ''

    # name of the table to retrieve patents
    table_for_retrieving = ''

    # table where new data is written
    table_for_storing_info = ''

    # Location of gensim model that already has embedding matrix or location to save a new embedding matrix
    model_location = ''

    # gensim_model to be used
    gensim_model = None

    # Gensim_Maker object to be used
    gensim_maker_obj = None

    # the maximum number of words any one patent's text is
    max_num_words = 0

    # dimension of any word vector in the model
    word_vector_dimension = 0

    def __init__(self, db_name, table_for_retrieving, model_location, table_for_storing_info, gensim_maker_obj=None,
                 need_to_make_table=False, gensim_model=None, need_to_create_models=True):
        self.db_name = db_name
        self.model_location = model_location
        # To protect against injection nonsense
        self.table_for_retrieving = self.remove_non_alphachars(table_for_retrieving)
        self.table_for_storing_info = self.remove_non_alphachars(table_for_storing_info)
        if gensim_maker_obj is None:
            self.gensim_maker_obj = GensimMaker(db_name, sql_table_name=table_for_retrieving, use_SQL_sentence_maker=True)
        else:
            self.gensim_maker_obj = gensim_maker_obj
        if need_to_create_models:
            self.gensim_maker_obj.generate_model()
            self.gensim_maker_obj.save_model(model_location)
        else:
            self.gensim_maker_obj.load_model(model_location)
        if gensim_model is None:
            self.gensim_model = self.gensim_maker_obj.get_model()

        if need_to_make_table:
            conn = sqlite3.connect(self.db_name)

            conn.execute('''CREATE TABLE ''' + self.table_for_storing_info + ''' (PATENT_ID TEXT NOT NULL, FIELD_DATA TEXT NOT NULL);''')
            conn.close()
            self.generate_organized_db()

        self.word_vector_dimension = self.gensim_maker_obj.get_dimension_of_a_word()

        self.max_num_words = self.find_max_num_words()

    # gets the maximum number of words in any file in the directory and make a list of company names
    def find_max_num_words(self):
        max_length = float('inf') * -1
        conn = sqlite3.connect(self.db_name)
        all_patents = conn.execute("SELECT * FROM " + self.table_for_storing_info + ";")
        for line in all_patents:
            if '#!%$^&' in str(line[1]):
                curr_length = len(str(line[1]).split('#!%$^&'))
            else:
                curr_length = len(str(line[1]).split())
            if curr_length > max_length:
                max_length = curr_length
        conn.close()
        return max_length

    def remove_non_alphachars(self, string_to_clean):
            return ''.join(char for char in string_to_clean if char.isalnum()or char is '_')

    # Takes the db and makes a new one with patents organized by patent name in one column and field value in the other
    def generate_organized_db(self):

        conn = sqlite3.connect(self.db_name)
        all_patents = conn.execute("SELECT * FROM " + self.table_for_retrieving)

        line_to_write = ''
        current_patent_id = 'PLACEHOLDER'
        for line in all_patents:
            patent_id = str(line[0])
            patent_field_value = str(line[1].replace('\n', ''))
            if current_patent_id == 'PLACEHOLDER':
                current_patent_id = patent_id
                line_to_write += patent_field_value
                continue
            if patent_id != current_patent_id:
                params = (current_patent_id, line_to_write)
                conn.execute("INSERT INTO " + self.table_for_storing_info + " VALUES (?, ?);", params)
                current_patent_id = patent_id
                line_to_write = patent_field_value
            else:
                current_patent_id = patent_id
                line_to_write += '#!%$^&' + patent_field_value
        # so last line is not ignored
        params = (current_patent_id, line_to_write)
        conn.execute(
                "INSERT INTO " + self.table_for_storing_info + " VALUES (?, ?);", params)
        conn.commit()
        conn.close()

    # Converts the row of the generated db to an array and pad it to the maximum length text string in the old table
    def convert_row_of_db_to_array(self, line_number):
        num_dim_of_a_word = self.gensim_maker_obj.get_dimension_of_a_word()
        conn = sqlite3.connect(self.db_name)
        params = (str(line_number),)
        query = conn.execute("SELECT * FROM " + self.table_for_storing_info + " LIMIT 1 OFFSET ?;", params)
        line = ''
        for row in query:
            line = str(row[1])
        big_array = np.zeros((num_dim_of_a_word, self.max_num_words))
        word_index = 0
        if '#!%$^&' in line:
            for word in line.split('#!%$^&'):
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
    def create_input_matrix(self, list_of_line_numbers):
        matrix_to_return = np.empty((len(list_of_line_numbers), self.gensim_maker_obj.get_dimension_of_a_word(), self.max_num_words, 1))
        matrix_depth_index = 0
        for line_number in list_of_line_numbers:
            matrix_to_add = self.convert_row_of_db_to_array(line_number)
            matrix_to_return[matrix_depth_index, :, :, 0] = matrix_to_add
            matrix_depth_index += 1
        matrix_to_return.view('float32')
        return matrix_to_return

    # returns a list of training set, validation set, and test set as 4d numpy arrays as a list in the following order:
    # training input, validation input, test input.
    # input is a list of 3 numbers which are the ratios of the training, validation, and test set, in order
    def get_input_set(self, ratios_list):
        random_line_numbers_list = self.get_random_line_numbers(ratios_list)
        return self.get_input_set_given_line_numbers(random_line_numbers_list)

    def get_input_set_given_line_numbers(self, list_of_line_numbers):

        train_set_lines_list = list_of_line_numbers[0]
        valid_set_lines_list = list_of_line_numbers[1]
        test_set_lines_list = list_of_line_numbers[2]

        return [self.create_input_matrix(train_set_lines_list),
                self.create_input_matrix(valid_set_lines_list),
                self.create_input_matrix(test_set_lines_list)]

    # given ratios for training, validation, and test data, returns 3 lists each containing random line numbers from
    # the organized table assuming line 0 is the first row
    def get_random_line_numbers(self, ratios_list):
        conn = sqlite3.connect(self.db_name)
        count_lines_in_file = conn.execute("SELECT COUNT(*) FROM " + self.table_for_storing_info + ";")
        num_lines_in_file = -1
        for num in count_lines_in_file:
            num_lines_in_file = num[0]
        num_lines_for_train_set = int(ratios_list[0] * num_lines_in_file)
        num_lines_for_valid_set = int(ratios_list[1] * num_lines_in_file)
        num_lines_for_test_set = num_lines_in_file - num_lines_for_train_set - num_lines_for_valid_set

        train_set_lines_list = list()
        valid_set_lines_list = list()
        test_set_lines_list = list()

        lines_list = list(range(0, num_lines_in_file))
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
        conn.close()

        return [train_set_lines_list, valid_set_lines_list, test_set_lines_list]

    # returns a list of patent ids given a list of line numbers in the organized patent file, assuming first row is
    # row 0
    def get_patent_ids_from_line_numbers(self, line_numbers_list):
        conn = sqlite3.connect(self.db_name)
        patent_id_list = list()

        for line in line_numbers_list:
            params = (str(line),)
            query = conn.execute(
                    "SELECT * FROM " + self.table_for_storing_info + " LIMIT 1 OFFSET ?;", params)
            for row in query:
                patent_id_list.append(str(row[0]))
        conn.close()
        return patent_id_list

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