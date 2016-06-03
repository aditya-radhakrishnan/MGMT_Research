import os, re
import sqlite3
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# class to iterate over sentences in a file, assuming the input is given as a patent + some property per line
class SQLSentenceMakerVersion2(object):

    # the location of the db that this class will iterate over
    db_name = ''

    # the name of the table that this class will iterate over
    table_name = ''

    # the name of the column where the information is stored
    column_name = ''

    # a flag that will turn every sentence to lower case if set to true
    lower_case_flag = False

    # a flag that will have the punctuation appear as words if set to true
    keep_punctuation_flag = True

    # the delimiter separating entries in the column values
    delimiter = ''

    # flag if the field data is just a text body with no delimiter
    is_text_body = False

    def __init__(self, db_name, table_name, column_name, delimiter = '', lower_case_flag = False, keep_punctuation_flag = True, is_text_body = True):
        self.db_name = db_name
        self.table_name = table_name
        self.column_name = column_name
        self.lower_case_flag = lower_case_flag
        self.keep_punctuation_flag = keep_punctuation_flag
        self.delimiter = delimiter
        self.is_text_body = is_text_body

    # iterator over the sentences in a column data base, splitting at punctuation followed by space which is assumed a sentence
    def __iter__(self):
        conn = sqlite3.connect(self.db_name)

        current_patent_id = -1
        current_patent_field_values = []
        all_sentences = conn.execute("SELECT " + self.column_name + " FROM " + self.table_name)
        if self.is_text_body:
            for row in all_sentences:
                line = str(row[0])
                if self.keep_punctuation_flag:
                    sentences = re.split('(.!?;)*', line)
                else:
                    sentences = re.split('[.!?;]*', line)
                for sentence in sentences:
                    sentence = self.filter(sentence)
                    yield sentence.split()
        else:
            for row in all_sentences:
                line = str(row[0])
                yield line.split(self.delimiter)
        conn.close()

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
        sentence = sentence.replace(' -', '')
        return sentence
