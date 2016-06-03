import os, re
import sqlite3

# class to iterate over sentences in a file, assuming the input is given as a patent + some property per line
class SQLSentenceMaker(object):

    # the location of the db that this class will iterate over
    db_name = ''

    # the name of the table that this class will iterate over
    table_name = ''

    # a flag that will turn every sentence to lower case if set to true
    lower_case_flag = False

    # a flag that will have the punctuation appear as words if set to true
    keep_punctuation_flag = True

    # a flag that says if the table fields are text bodies
    are_text_bodies = False

    def __init__(self, db_name, table_name, lower_case_flag = False, keep_punctuation_flag = True, text_bodies=False):
        self.db_name = db_name
        self.table_name = table_name
        self.lower_case_flag = lower_case_flag
        self.keep_punctuation_flag = keep_punctuation_flag
        self.are_text_bodies = text_bodies

    # iterator over the sentences in a data base, splitting at punctuation followed by space which is assumed a sentence
    def __iter__(self):
        conn = sqlite3.connect(self.db_name)

        current_patent_id = -1
        current_patent_field_values = []
        all_patents = conn.execute("SELECT * FROM " + self.table_name)
        if self.are_text_bodies:
            for patents in all_patents:
                line = str(patents[1])
                if self.keep_punctuation_flag:
                    sentences = re.split('[.!?;]*', line)
                else:
                    sentences = re.split('[.!?;]*', line)
                for sentence in sentences:
                    sentence = self.filter(sentence)
                    yield sentence.split()
        else:
            for line in all_patents:
                if self.lower_case_flag:
                    line = line.lower()
                patent_id = line[0]
                patent_field_value = str(line[1].replace('\n', ''))
                if patent_id != current_patent_id and current_patent_id != -1:
                    yield current_patent_field_values
                    current_patent_field_values = []
                    current_patent_id = patent_id
                    current_patent_field_values.append(patent_field_value)
                else:
                    current_patent_id = patent_id
                    current_patent_field_values.append(patent_field_value)
            yield current_patent_field_values
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
