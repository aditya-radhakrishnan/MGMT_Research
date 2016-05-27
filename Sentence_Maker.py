import os, re

# class to iterate over sentences in all files in a directory
class SentenceMaker(object):
    # the location of the directory that this class will iterate over
    file_name = ''
    # a flag that will turn every sentence to lower case if set to true
    lower_case_flag = False
    # a flag that will have the punctuation appear as words if set to true
    keep_punctuation_flag = True

    def __init__(self, file_name, lower_case_flag = False, keep_punctuation_flag = True):
        self.file_name = file_name
        self.lower_case_flag = lower_case_flag
        self.keep_punctuation_flag = keep_punctuation_flag

    # iterator over the sentences in a file, splitting at punctuation followed by space which is assumed a sentence
    def __iter__(self):
        for line in open(self.file_name):
            if self.lower_case_flag:
                line = line.lower()
            if self.keep_punctuation_flag:
                sentences = re.split('(.!?;)*', line)
            else:
                sentences = re.split('[.!?;]*', line)
            for sentence in sentences:
                sentence = self.filter(sentence)
                yield sentence.split()

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
