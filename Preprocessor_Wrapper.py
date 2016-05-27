from Preprocessor import Preprocessor

class PreprocessorWrapper(object):

    # list of internal preprocessors that are used
    preprocessor_list = list()

    # the desired ratios of how much is training data, how much is validation data, how much is test data in that order
    ratios_list = list()

    # the random line numbers to grab from every file
    line_number_list = list()

    # list of files is the list of files that will be input into the stacks, one file per stack
    # list of models is the list of the locations where the word2vec models will be stored for each file
    # ratios list must be a list of 3 numbers with the first being how the document is being divided into training, validation, and test data respectively
    # the third parameter can be set to false if the models have already been saved previously and will be loaded from the list of models location
    def __init__(self, list_of_files, list_of_models, ratios_list, need_to_make_models = True):
        if len(list_of_files) != len(list_of_models):
            raise ValueError("list of files must be same length as list of models as each file needs its own model")
        for index in range(len(list_of_files)):
            preprocessor_to_add = Preprocessor(list_of_files[index], list_of_models[index], need_to_create_model = need_to_make_models)
            self.preprocessor_list.append(preprocessor_to_add)
        self.ratios_list = ratios_list
        self.line_number_list = self.preprocessor_list[0].get_random_line_numbers(self.ratios_list)

    # get the training data labels
    def get_training_data_labels(self):
        return self.preprocessor_list[0].get_data_set_given_line_numbers(self.line_number_list)[1]

    def get_validation_data_labels(self):
        return self.preprocessor_list[0].get_data_set_given_line_numbers(self.line_number_list)[3]

    def get_test_data_labels(self):
        return self.preprocessor_list[0].get_data_set_given_line_numbers(self.line_number_list)[5]

    # channel_num 0 refers to the first stack, 1 refers to the second stack, 2 refers to the third
    def get_training_data_from_channel(self, channel_num):
        return self.preprocessor_list[channel_num].get_data_set_given_line_numbers(self.line_number_list)[0]

    def get_validation_data_from_channel(self, channel_num):
        return self.preprocessor_list[channel_num].get_data_set_given_line_numbers(self.line_number_list)[2]

    def get_test_data_from_channel(self, channel_num):
        return self.preprocessor_list[channel_num].get_data_set_given_line_numbers(self.line_number_list)[4]

    def get_first_preprocessor(self):
        return self.preprocessor_list[0]
# EXAMPLES TO USE THE CODE

#file_location_list = ['/home/trehuang/Desktop/Earnings Calls Data/Split_Documents/part1.txt',
#                      '/home/trehuang/Desktop/Earnings Calls Data/Split_Documents/part2.txt',
#                      '/home/trehuang/Desktop/Earnings Calls Data/Split_Documents/part3.txt']

#model_location_list = ['/home/trehuang/Desktop/Earnings Calls Data/Split_Documents/model1.txt',
#                      '/home/trehuang/Desktop/Earnings Calls Data/Split_Documents/model2.txt',
#                      '/home/trehuang/Desktop/Earnings Calls Data/Split_Documents/model3.txt']

#ratios = [.5, .3, .2]

# preprocessor_wrapper = PreprocessorWrapper(file_location_list, model_location_list, ratios)

# this gets the training data as a 4D numpy array from the second file to go into stack number 2
# preprocessor_wrapper.get_training_data_from_channel(1)

# this gets the training labels as a 2D numpy array
# preprocessor_wrapper.get_training_data_labels()
