from SQL_Preprocessor import SQLPreprocessor
from Forward_Citations_Generator import ForwardCitationsGenerator
from Gensim_Maker import GensimMaker
import sqlite3
import numpy as np

class SQLPreprocessorWrapper(object):

    # the name of the database where tables are obtained
    db_name = ''

    # list of internal preprocessors that are used
    preprocessor_list = list()

    # the desired ratios of how much is training data, how much is validation data, how much is test data in that order
    ratios_list = list()

    # the random patents to grab from every table
    line_numbers_list = list()

    # the dictionary of a patent id to its classifcation as a one-hot vector
    classification_dict = dict()

    # one hot dict size
    one_hot_dict_size = 0

    # the dictionary of a patent id to its number of forward citations
    forward_citation_dict = dict()

    # db name is the name of the database where the tables are retrieved
    # list of tables for retrieval is the list of the table names where data will be retrieved
    # list of models is the list of the locations where the word2vec models will be stored for each file
    # list of tables_for_writing are the tables to write output to
    # ratios list must be a list of 3 numbers with the first being how the document is being divided into training, validation, and test data respectively
    # list of gensim_makers is the list of different gensim makers that can each have their own parameters
    # flags are self explanatory
    def __init__(self, db_name, list_of_tables_for_retrieval, list_of_tables_for_writing, list_of_models, ratios_list,
                 list_of_gensim_makers = None, need_to_make_models_bool_list = [], need_to_make_tables_bool_list = []):
        self.db_name = db_name
        if len(list_of_tables_for_retrieval) != len(list_of_models) or len(list_of_tables_for_retrieval) != len(list_of_tables_for_writing):
            raise ValueError("list of tables must be same length as list of models as each table needs its own model")
        if list_of_gensim_makers is None:
            for index in range(len(list_of_tables_for_retrieval)):
                preprocessor_to_add = SQLPreprocessor(db_name, list_of_tables_for_retrieval[index],
                                                      list_of_models[index],
                                                      list_of_tables_for_writing[index],
                                                      need_to_make_table=need_to_make_tables_bool_list[index],
                                                      need_to_create_models=need_to_make_models_bool_list[index])
                self.preprocessor_list.append(preprocessor_to_add)
        else:
            for index in range(len(list_of_tables_for_retrieval)):
                preprocessor_to_add = SQLPreprocessor(db_name, list_of_tables_for_retrieval[index],
                                                      list_of_models[index],
                                                      list_of_tables_for_writing[index],
                                                      gensim_maker_obj=list_of_gensim_makers[index],
                                                      need_to_make_table=need_to_make_tables_bool_list[index],
                                                      need_to_create_models=need_to_make_models_bool_list[index])
                self.preprocessor_list.append(preprocessor_to_add)
        self.ratios_list = ratios_list
        self.line_numbers_list = self.preprocessor_list[0].get_random_line_numbers(self.ratios_list)

    def get_preprocessor(self, processor_num):
        return self.preprocessor_list[processor_num]

    def get_dimension_of_word_from_preprocessor(self, processor_num):
        return self.preprocessor_list[processor_num].word_vector_dimension

    def get_maximum_number_of_words_in_a_sentence(self, processor_num):
        return self.preprocessor_list[processor_num].max_num_words

    # channel_num 0 refers to the first stack, 1 refers to the second stack, 2 refers to the third
    def get_training_input_from_channel(self, channel_num):
        return self.preprocessor_list[channel_num].get_input_set_given_line_numbers(self.line_numbers_list)[0]

    def get_validation_input_from_channel(self, channel_num):
        return self.preprocessor_list[channel_num].get_input_set_given_line_numbers(self.line_numbers_list)[1]

    def get_testing_input_from_channel(self, channel_num):
        return self.preprocessor_list[channel_num].get_input_set_given_line_numbers(self.line_numbers_list)[2]

    def make_forward_citation_table(self, backward_citation_table, citation_column_name, new_table_name):
        self.forward_citation_generator = ForwardCitationsGenerator(self.db_name, backward_citation_table,
                                                                    citation_column_name, new_table_name)

    def make_classification_dict(self, classification_table):
        conn = sqlite3.connect(self.db_name)
        query = conn.execute("SELECT * FROM " + classification_table)
        classification_list = list()
        patent_id_to_classifcation = dict()

        for row in query:
            if str(row[4]) not in classification_list:
                classification_list.append(str(row[4]))
            patent_id_to_classifcation[str(row[0])] = str(row[4])

        conn.close()

        classification_to_one_hot_dict = self.create_one_hot_encoding_dictionary(classification_list)

        for patent in patent_id_to_classifcation.keys():
            self.classification_dict[patent] = classification_to_one_hot_dict[patent_id_to_classifcation[patent]]

    # makes the forward citation dictionary
    def make_forward_citation_dict(self, back_citation_table_name, back_citation_column_name, new_table_name):
        forward_citation_obj = ForwardCitationsGenerator(self.db_name, back_citation_table_name, back_citation_column_name, new_table_name)
        conn = sqlite3.connect(self.db_name)
        query = conn.execute(("SELECT * FROM " + new_table_name))
        # each row is going to be patent id | number of citations
        for row in query:
            self.classification_dict[str(row[0])] = str(row[1])

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
        self.one_hot_dict_size = len(dictionary)
        return dictionary

    # creates the correct answer matrix by vertically stacking the classification vectors of the input list of answers
    def create_answer_matrix(self, list_of_patent_ids):
        matrix_to_return = np.empty((len(list_of_patent_ids), self.one_hot_dict_size))
        matrix_row_index = 0
        for patent in list_of_patent_ids:
            if patent in self.classification_dict:
                matrix_to_return[matrix_row_index, :] = self.classification_dict[patent]
            matrix_row_index += 1
        matrix_to_return.view('float32')
        return matrix_to_return

    # returns the training input and training labels matrices in a list of 2 items in respective order
    def get_training_data_from_channel(self, channel_num):
        training_input = self.get_training_input_from_channel(channel_num)
        answer_matrix = self.get_training_labels()
        return [training_input, answer_matrix]

    # returns the validation input and validation labels matrices in a list of 2 items in respective order
    def get_validation_data_from_channel(self, channel_num):
        validation_input = self.get_validation_input_from_channel(channel_num)
        answer_matrix = self.get_validation_labels()
        return [validation_input, answer_matrix]

    # returns the testing input and testing labels matrices in a list of 2 items in respective order
    def get_testing_data_from_channel(self, channel_num):
        testing_input = self.get_testing_input_from_channel(channel_num)
        answer_matrix = self.get_testing_labels()
        return [testing_input, answer_matrix]

    # returns the training labels
    def get_training_labels(self):
        list_of_patent_ids_of_labels = self.preprocessor_list[0].get_patent_ids_from_line_numbers(
            self.line_numbers_list[0])
        answer_matrix = self.create_answer_matrix(list_of_patent_ids_of_labels)
        return answer_matrix

    # returns the validation labels
    def get_validation_labels(self):
        list_of_patent_ids_of_labels = self.preprocessor_list[0].get_patent_ids_from_line_numbers(
            self.line_numbers_list[1])
        answer_matrix = self.create_answer_matrix(list_of_patent_ids_of_labels)
        return answer_matrix

    # returns the testing labels
    def get_testing_labels(self):
        list_of_patent_ids_of_labels = self.preprocessor_list[0].get_patent_ids_from_line_numbers(
            self.line_numbers_list[2])
        answer_matrix = self.create_answer_matrix(list_of_patent_ids_of_labels)
        return answer_matrix
'''
# EXAMPLES TO USE THE CODE
db_name = 'patent_database.db'

list_of_tables_for_retrieval = ['abstracts',
                                'inventors']

model_location_list = ['/home/trehuang/Desktop/model_for_abstracts.txt',
                       '/home/trehuang/Desktop/model_for_inventors.txt']

list_of_tables_for_writing =['organized_abstracts',
                             'organized_inventors']

# Lets us customize the parameters for each stack for the gensim models
gensim_maker_list = [GensimMaker(db_name, min_to_ignore=4, size=60, sql_table_name=list_of_tables_for_retrieval[0], use_SQL_sentence_maker=True, use_SQL_sentence_maker_for_texts=True),
                     GensimMaker(db_name, min_to_ignore=2, size=30, sql_table_name=list_of_tables_for_retrieval[1], use_SQL_sentence_maker=True)]

ratios = [.7, .15, .15]

# same as before, you need to make the models if running for the first time
make_models = True

# If the tables have not been made before, this MUST BE TRUE. If the tables have been made before, this MUST BE FALSE
make_tables = True

preprocessor_wrapper = SQLPreprocessorWrapper(db_name,
                                              list_of_tables_for_retrieval,
                                              list_of_tables_for_writing,
                                              model_location_list, ratios,
                                              gensim_maker_list,
                                              need_to_make_models=make_models,
                                              need_to_make_tables=make_tables)

preprocessor_wrapper.make_classification_dict('classifications')
testing_data, testing_labels = preprocessor_wrapper.get_testing_data_from_channel(1)
dimension_of_a_word_from_first_channel = preprocessor_wrapper.get_dimension_of_word_from_preprocessor(0)
max_number_of_words_from_second_stack = preprocessor_wrapper.get_maximum_number_of_words_in_a_sentence(1)
'''