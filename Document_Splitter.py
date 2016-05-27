import os
import re

directory_to_write_to = '/home/trehuang/Desktop/Earnings Calls Data/Split_Documents'
directory_of_files_to_split = '/home/trehuang/Desktop/Earnings Calls Data/All_Data'

first_document_to_write_to = open(directory_to_write_to + '/part1.txt', 'w')
second_document_to_write_to = open(directory_to_write_to + '/part2.txt', 'w')
third_document_to_write_to = open(directory_to_write_to + '/part3.txt', 'w')

words_of_text_body_we_want = 500
max_num_firms = 50
firm_set = set()

for file_name in os.listdir(directory_of_files_to_split):
    for line in open(os.path.join(directory_of_files_to_split, file_name)):
        split_line_list = re.split('\t*', line)

        company_name = split_line_list[1]
        firm_set.add(company_name)

        if len(firm_set) > max_num_firms:
            break

        text_body = split_line_list[-1]
        words_of_text_body = text_body.split()

        try:
            first_document_text_body = ' '.join(words_of_text_body[:words_of_text_body_we_want])
        except IndexError:
            first_document_text_body = 'Data not found'
        try:
            second_document_text_body = ' '.join(words_of_text_body[words_of_text_body_we_want:(2*words_of_text_body_we_want)])
        except IndexError:
            second_document_text_body = 'Data not found'
        try:
            third_document_text_body = ' '.join(words_of_text_body[(words_of_text_body_we_want*2):(3*words_of_text_body_we_want)])
        except IndexError:
            third_document_text_body = 'Data not found'

        first_document_to_write_to.write(company_name + '\t' + first_document_text_body + '\n')
        second_document_to_write_to.write(company_name + '\t' + second_document_text_body + '\n')
        third_document_to_write_to.write(company_name + '\t' + third_document_text_body + '\n')
