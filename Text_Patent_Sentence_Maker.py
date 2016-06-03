import os, re

# class to iterate over sentences in all files in a directory
class TextPatentSentenceMaker(object):
    # the location of the directory that this class will iterate over
    file_name = ''
    # a flag that will turn every sentence to lower case if set to true
    lower_case_flag = False

    def __init__(self, file_name, lower_case_flag = False):
        self.file_name = file_name
        self.lower_case_flag = lower_case_flag

    # iterator over the sentences in a file assuming a sentence is all the fields values belonging to a particular patent
    def __iter__(self):
        current_patent_id = -1
        current_patent_field_values = []
        for line in open(self.file_name):
            if self.lower_case_flag:
                line = line.lower()
            split_line = line.split('\t')
            patent_id = split_line[0]
            patent_field_value = split_line[1].replace('\n', '')
            if patent_id != current_patent_id and current_patent_id != -1:
                yield current_patent_field_values
                current_patent_field_values = []
                current_patent_id = patent_id
                current_patent_field_values.append(patent_field_value)
            else:
                current_patent_id = patent_id
                current_patent_field_values.append(patent_field_value)
        yield current_patent_field_values

    # filters a sentence so that only words remain by stripping ,:;$%&'"-
    def filter(self, sentence):
        sentence = sentence.replace(',', '')
        sentence = sentence.replace(':', '')
        sentence = sentence.replace(';', '')
        sentence = sentence.replace('$', '')
        sentence = sentence.replace('%', '')
        sentence = sentence.replace('&', '')
        sentence = sentence.replace('\'', '')
        sentence = sentence.replace('\"', '')
        sentence = sentence.replace('-', '')
        return sentence