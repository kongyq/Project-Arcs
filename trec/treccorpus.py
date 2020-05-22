import logging
# import multiprocessing

from os import listdir
from os.path import isfile, join
from os import linesep

import re

# from gensim import utils

from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.dictionary import Dictionary

logger = logging.getLogger(__name__)

TOKEN_MIN_LEN = 2
TOKEN_MAX_LEN = 15


def strip_tags(buffer, start_pos):
    """
    Strip all irrelevant tags
    :param buffer: text content
    :param start_pos: start position
    :return: cleaned text content
    """
    if start_pos > 0:
        buffer = buffer[start_pos:]
    return re.sub(pattern=r"<[^>]*>", repl=" ", string=buffer)


def extract(buffer, start_tag, end_tag, max_pos, noise_prefix):
    """
    extract content from specific tag name
    :param buffer: text content
    :param start_tag: start tag
    :param end_tag: end tag
    :param max_pos: max position extraction can reach
    :param noise_prefix: noise prefix to be discarded
    :return: extract content from original content
    """
    k1 = buffer.find(start_tag)
    if k1 >= 0 and (max_pos < 0 or k1 < max_pos):
        k1 += len(start_tag)
        k2 = buffer.find(end_tag, k1)
        if k2 >= 0 and (max_pos < 0 or k2 < max_pos):
            if noise_prefix is not None:
                noise_prefix_size = noise_prefix.size()
                for i in range(0, noise_prefix_size):
                    prefix = noise_prefix[i]
                    k1a = buffer.find(prefix, k1)
                    if 0 <= k1a < k2:
                        k1 = k1a + len(prefix)

            return buffer[k1:k2].strip()

    return None


class TrecCorpus(TextCorpus):

    def __init__(self, pname, dictionary=None, metadata=True):
        # super(TrecCorpus, self).__init__()
        self.test = True
        self.pname = pname
        self.metadata = True
        if dictionary is None:
            self.dictionary = Dictionary(self.get_texts())
        else:
            self.dictionary = dictionary

        self.file_list = [file for file in listdir(self.pname) if isfile(join(self.pname, file))]
        self.reader = None
        self.file_pointer = 0
        super().__init__()

    def open_next_file(self):
        if self.reader is not None:
            self.reader.close()
        if self.file_pointer < len(self.file_list):
            self.reader = open(self.file_list[self.file_pointer], 'r')
            self.file_pointer += 1
        else:
            self.reader = None

    def read(self, buffer, line_start, collect_match_line, collect_all):
        sep = ""
        while True:
            while True:
                curr_line = self.reader.readLine()
                if curr_line is None:
                    self.open_next_file()
                else:
                    if line_start is not None and curr_line.startswith(line_start):
                        if collect_match_line:
                            buffer = buffer + sep + curr_line
                            sep = linesep
                        return

                    if collect_all:
                        buffer = buffer + sep + curr_line
                        sep = linesep

    def trec_doc_parser(self, buffer):
        # TEXT = "<TEXT>"
        # TEXT_END = "</TEXT>"

        mark = 0
        h1 = buffer.find("<TEXT>")
        if h1 >= 0:
            h2 = buffer.find("</TEXT>", h1)
            mark = h1 + len("<TEXT>")
        return strip_tags(buffer, mark)

    def get_texts(self):
        if self.reader is None:
            self.open_next_file()

        doc_buffer = ""
        self.read(doc_buffer, "<DOC>", collect_match_line=False, collect_all=False)
        doc_buffer = ""
        self.read(doc_buffer, "<DOCNO>", collect_match_line=True, collect_all=False)

        docno_value = doc_buffer[len("<DOCNO>"): doc_buffer.find("</DOCNO>", len("<DOCNO>"))].strip()

        doc_buffer = ""
        self.read(doc_buffer, "</DOC>", collect_match_line=False, collect_all=True)

        return self.trec_doc_parser(doc_buffer), docno_value
