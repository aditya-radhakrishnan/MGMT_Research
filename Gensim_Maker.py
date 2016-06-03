import gensim
import logging
import warnings
from Sentence_Maker import SentenceMaker
from Text_Patent_Sentence_Maker import TextPatentSentenceMaker
from SQL_Sentence_Maker_Version2 import SQLSentenceMakerVersion2

class GensimMaker(object):

    # remove the FutureWarning that regex keeps giving
    warnings.simplefilter(action = "ignore", category = FutureWarning)

    # logging format configuration
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # location of the file where we will gather all of our words, or the database name in case of sql sentence maker
    location = ''

    # the name of the table of the sql table to gather words from
    table_name = ''

    # the name of the column from the table to gather words from
    column_name = ''

    # the minimum number of occurrences a word must have to not be ignored. Words occurring less than the number are thrown out
    min_to_ignore = 0

    # the number of dimensions for each word vector. Default is 100 but can be increased on bigger datasets
    size = 0

    # must have cython installed to use. It is the number of threads working on the data.
    threads = 0

    # This makes gensim run on skipgram if 1
    sg = 0

    # the model which contains the dictionary of every word to its vector representation
    model = None

    # tells the class to use the SQL sentence maker instead of standard sentence maker
    use_sql_sentence_maker = False

    # tells the class to use the patent sentence maker instead of standard sentence maker
    use_patent_sentence_maker = False

    # tells the class to use the SQL sentence maker where every patent has only one field
    use_sql_sentence_maker_for_text = False

    # the delimiter to be used for SQL databases
    delimiter = ''

    def __init__(self, location, min_to_ignore = 1, size = 20, threads = 4, sg = 1, model = None, sql_table_name = '', column_name = '', delimiter = '',
                 use_SQL_sentence_maker = False, use_SQL_sentence_maker_for_texts=False, use_patent_sentence_maker = True):
        self.location = location
        self.min_to_ignore = min_to_ignore
        self.size = size
        self.threads = threads
        self.sg = sg
        self.model = model
        self.table_name = sql_table_name
        self.use_sql_sentence_maker = use_SQL_sentence_maker
        self.use_patent_sentence_maker = use_patent_sentence_maker
        self.use_sql_sentence_maker_for_text = use_SQL_sentence_maker_for_texts
        self.delimiter = delimiter
        self.column_name = column_name

    # creates the model by iterating over every sentence in every document in the directory specified by location
    def generate_model(self):
        if self.use_sql_sentence_maker:
            if self.use_sql_sentence_maker_for_text:
                sentences = SQLSentenceMakerVersion2(self.location, self.table_name, self.column_name, delimiter=self.delimiter, is_text_body=True)
            else:
                sentences = SQLSentenceMakerVersion2(self.location, self.table_name, self.column_name, delimiter=self.delimiter, is_text_body=False)
        elif self.use_patent_sentence_maker:
            sentences = TextPatentSentenceMaker(self.location)
        else:
            sentences = SentenceMaker(self.location)
        self.model = gensim.models.Word2Vec(sentences, min_count = self.min_to_ignore, size = self.size, workers = self.threads, sg = self.sg)

    def save_model(self, location_to_save_model):
        self.model.save(location_to_save_model)

    def load_model(self, location_to_load_model):
        self.model = gensim.models.Word2Vec.load(location_to_load_model)

    def get_model(self):
        return self.model

    def get_embedding_matrix(self):
        return self.model.syn0

    # returns the vector representation of the word as given by the model in a float32 numpy array
    def get_vector_of_word(self, word):
        return self.model[word]

    def get_dimension_of_a_word(self):
        return self.size