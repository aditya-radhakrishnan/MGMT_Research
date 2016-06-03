from SQL_Preprocessor_Version2 import SQLPreprocessorVersion2
from Forward_Citations_Generator_Version2 import ForwardCitationsGeneratorVersion2
import sqlite3
import numpy as np
from Gensim_Maker import GensimMaker

class SQLPreprocessorWrapperVersion2(object):

    # the name of the database where tables are obtained
    db_name = ''

    # list of internal preprocessors that are used
    preprocessor_list = list()

    # the desired ratios of how much is training data, how much is validation data, how much is test data in that order
    ratios_list = list()

    # the name of the table
    table_name = ''

    # the random patents to grab from every table
    random_patents_list = list()

    # temporary forward citations dict
    forward_citation_dict = dict()

    # the dictionary of a patent id to its classifcation as a one-hot vector
    classification_dict = dict()

    # one hot dict size
    one_hot_dict_size = 0

    # temporary renewal dict
    renewal_dict = dict()

    # the delimiter used
    delimiter = ''

    def __init__(self, db_name, table_name, list_of_column_names, list_of_models, ratios_list, delimiter,
                 list_of_gensim_makers = None, need_to_make_models_bool_list = []):
        self.db_name = db_name
        self.delimiter = delimiter
        self.table_name = self.remove_non_alphachars(table_name)
        if len(list_of_column_names) != len(list_of_models):
            raise ValueError("list of tables must be same length as list of models as each table needs its own model")
        if list_of_gensim_makers is None:
            for index in range(len(list_of_column_names)):
                preprocessor_to_add = SQLPreprocessorVersion2(db_name,
                                                              table_name,
                                                              list_of_models[index],
                                                              list_of_column_names[index],
                                                              need_to_create_models=need_to_make_models_bool_list[index],
                                                              delimiter=delimiter)
                self.preprocessor_list.append(preprocessor_to_add)
        else:
            for index in range(len(list_of_column_names)):
                preprocessor_to_add = SQLPreprocessorVersion2(db_name,
                                                              table_name,
                                                              list_of_models[index],
                                                              list_of_column_names[index],
                                                              need_to_create_models=need_to_make_models_bool_list[index],
                                                              gensim_maker_obj=list_of_gensim_makers[index],
                                                              delimiter = delimiter)
                self.preprocessor_list.append(preprocessor_to_add)
        self.ratios_list = ratios_list
        self.random_patents_list = self.preprocessor_list[0].get_random_patent_ids(self.ratios_list)

    def get_preprocessor(self, processor_num):
        return self.preprocessor_list[processor_num]

    def get_dimension_of_word_from_preprocessor(self, processor_num):
        return self.preprocessor_list[processor_num].word_vector_dimension

    def get_maximum_number_of_words_in_a_sentence(self, processor_num):
        return self.preprocessor_list[processor_num].max_num_words

    # channel_num 0 refers to the first stack, 1 refers to the second stack, 2 refers to the third
    def get_training_input_from_channel(self, channel_num):
        return self.preprocessor_list[channel_num].get_input_set_given_patent_ids(self.random_patents_list)[0]

    def get_validation_input_from_channel(self, channel_num):
        return self.preprocessor_list[channel_num].get_input_set_given_patent_ids(self.random_patents_list)[1]

    def get_testing_input_from_channel(self, channel_num):
        return self.preprocessor_list[channel_num].get_input_set_given_patent_ids(self.random_patents_list)[2]

    def make_classification_dict(self, classification_column_name):
        conn = sqlite3.connect(self.db_name)
        classification_column_name = self.remove_non_alphachars(classification_column_name)
        query = conn.execute("SELECT patent_id, " + classification_column_name + " FROM " + self.table_name)
        classification_list = list()
        patent_id_to_classifcation = dict()

        for row in query:
            if str(row[1]) not in classification_list:
                classification_list.append(str(row[1]))
            patent_id_to_classifcation[str(row[0])] = str(row[1])

        conn.close()

        classification_to_one_hot_dict = self.create_one_hot_encoding_dictionary(classification_list)

        for patent in patent_id_to_classifcation.keys():
            self.classification_dict[patent] = classification_to_one_hot_dict[patent_id_to_classifcation[patent]]

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

    # makes a patent to number of forward citation dictionary
    def make_forward_citation_dict(self, back_citation_table_name, back_citation_column_name, new_table_name):
        _ = ForwardCitationsGeneratorVersion2(self.db_name, back_citation_table_name,
                                                                 back_citation_column_name, new_table_name)
        conn = sqlite3.connect(self.db_name)
        query = conn.execute(("SELECT * FROM " + new_table_name))
        forward_citation_dict = dict()

        # each row is going to be patent id | number of citations
        for row in query:
            forward_citation_dict[str(row[0])] = str(row[1])
        conn.close()
        return forward_citation_dict

    # makes a patent to number of renewals dictionary
    def make_renewal_dict(self, renewal_table_name):
        conn = sqlite3.connect(self.db_name)
        query = conn.execute(("SELECT * FROM " + renewal_table_name))
        renewal_dict = dict()

        # each row is going to be patent id | number of citations
        for row in query:
            renewal_dict[str(row[0])] = str(row[1])
        conn.close()
        return renewal_dict

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

    '''
    # TODO
    # creates the correct answer matrix by vertically stacking the concatenated citation and renewal vectors of the input list of answers
    def create_answer_matrix(self, list_of_patent_ids):
        matrix_to_return = np.empty((len(list_of_patent_ids), 2))
        matrix_row_index = 0
        for patent in list_of_patent_ids:
            if patent in self.classification_dict:
                matrix_to_return[matrix_row_index, :] = np.asarray([self.forward_citation_dict[patent], self.renewal_dict[patent]])
            matrix_row_index += 1
        matrix_to_return.view('float32')
        return matrix_to_return
    '''

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
        answer_matrix = self.create_answer_matrix(self.random_patents_list[0])
        return answer_matrix

    # returns the validation labels
    def get_validation_labels(self):
        answer_matrix = self.create_answer_matrix(self.random_patents_list[1])
        return answer_matrix

    # returns the testing labels
    def get_testing_labels(self):
        answer_matrix = self.create_answer_matrix(self.random_patents_list[2])
        return answer_matrix

    def remove_non_alphachars(self, string_to_clean):
        return ''.join(char for char in string_to_clean if char.isalnum() or char is '_')

'''
# EXAMPLES TO USE THE CODE
db_name = 'patent_database.db'

table_name = 'patent_info'

delimiter = '#!%^'

list_of_columns_for_retrieval = ['abstract',
                                'inventors']

model_location_list = ['/home/trehuang/Desktop/model_for_abstracts.txt',
                       '/home/trehuang/Desktop/model_for_inventors.txt']

# Lets us customize the parameters for each stack for the gensim models
gensim_maker_list = [GensimMaker(db_name, min_to_ignore=4, size=60, sql_table_name=table_name,
                                 column_name = list_of_columns_for_retrieval[0], delimiter = delimiter,
                                 use_SQL_sentence_maker=True, use_SQL_sentence_maker_for_texts=True),
                     GensimMaker(db_name, min_to_ignore=2, size=20, sql_table_name=table_name,
                                 column_name=list_of_columns_for_retrieval[1], delimiter=delimiter,
                                 use_SQL_sentence_maker=True, use_SQL_sentence_maker_for_texts=True)]

ratios = [.7, .15, .15]

# same as before, you need to make the models if running for the first time
make_models = [False, False]

preprocessor_wrapper = SQLPreprocessorWrapperVersion2(db_name,
                                                      table_name,
                                                      list_of_columns_for_retrieval,
                                                      model_location_list,
                                                      ratios,
                                                      delimiter,
                                                      gensim_maker_list,
                                                      need_to_make_models_bool_list=make_models)

preprocessor_wrapper.make_classification_dict('main_cpc_class')
testing_data, testing_labels = preprocessor_wrapper.get_testing_data_from_channel(1)
dimension_of_a_word_from_first_channel = preprocessor_wrapper.get_dimension_of_word_from_preprocessor(0)
max_number_of_words_from_second_stack = preprocessor_wrapper.get_maximum_number_of_words_in_a_sentence(1)
'''