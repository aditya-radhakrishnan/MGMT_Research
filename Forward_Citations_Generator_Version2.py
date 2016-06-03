import sqlite3

# generates a table with column 1 being patent ID and column 2 being the number of forward citations that patent has
class ForwardCitationsGeneratorVersion2(object):

    # the name of the database to access
    db_name = ''

    # the name of the table where the data is held
    table_name = ''

    # the name of the column header for the backward citations
    citation_column_name = ''

    # the name of the new table where the forward citations will go
    new_table_name = ''

    def __init__(self, database_name, table_name, citation_column_name, new_table_name):
        self.db_name = database_name
        # prevent any injection nonsense
        self.table_name = self.remove_non_alphachars(table_name)
        self.citation_column_name = self.remove_non_alphachars(citation_column_name)
        self.new_table_name = self.remove_non_alphachars(new_table_name)

        # create the new table
        conn = sqlite3.connect(self.db_name)
        conn.execute9("DROP TABLE IF EXISTS " + self.new_table_name)
        conn.execute(
            '''CREATE TABLE ''' + self.new_table_name + ''' (patent_id TEXT NOT NULL, num_forward_citations TEXT NOT NULL);''')

        self.generate_new_table()

    # generates the new table and writes it to the database
    def generate_new_table(self):
        conn = sqlite3.connect(self.db_name)
        patent_ids = self.get_all_patent_ids()

        # basically for each patent we count the number of times it appears as a backward citation in the old table using sql calls
        for patent_id in patent_ids:
            query = conn.execute("SELECT COUNT(*) FROM " + self.table_name + " WHERE " + self.citation_column_name + " LIKE \"%" + patent_id + "%\"")
            num_citations = -1
            for value in query:
                num_citations = value[0]
            # now we write to the new table
            conn.execute("INSERT INTO " + self.new_table_name + " VALUES(" + patent_id + "," + str(num_citations) + ");")

        # save the new table
        conn.commit()

    # makes a list of all the patent IDs to write for the new table
    def get_all_patent_ids(self):
        output_list = list()
        conn = sqlite3.connect(self.db_name)
        all_patents = conn.execute("SELECT * FROM " + self.table_name + ";")

        for row in all_patents:
            if str(row[0]) not in output_list:
                output_list.append(str(row[0]))

        return output_list

    # strips characters that aren't a-zA-Z0-9
    def remove_non_alphachars(self, string_to_clean):
        return ''.join(char for char in string_to_clean if char.isalnum() or char is '_')

    def get_num_forward_citations(self, patent_id):
        conn = sqlite3.connect(self.db_name)
        query = conn.execute("SELECT * FROM " + self.new_table_name + " WHERE patent_id=" + patent_id)
        for row in query:
            return str(row[1])