import logging
# import multiprocessing

from os import listdir
from os.path import isfile, join
from os import linesep

import re

from gensim import utils

from gensim.corpora.textcorpus import TextDirectoryCorpus
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


class TrecCorpus(TextDirectoryCorpus):

    def __init__(self, input, dictionary=None, metadata=True,
                 lines_are_documents=False, min_depth=0, max_depth=None, **kwargs):

        super(TrecCorpus, self).__init__(input=input,
                                         dictionary=dictionary,
                                         metadata=metadata,
                                         lines_are_documents=lines_are_documents,
                                         min_depth=min_depth, max_depth=max_depth, **kwargs)

    def get_texts(self):

        for file in self.getstream():
            for docno, text in self.parse_file(file):
                if self.metadata:
                    yield utils.tokenize(self.parse_text(text), lowercase=True), (docno,)
                else:
                    yield utils.tokenize(self.parse_text(text), lowercase=True)


    def parse_file(self, file):
        while file is not None:
            self.read_doc(file, "<DOC>", collect_match_line=False, collect_all=False)

            docno = self.read_doc(file, "<DOCNO>", collect_match_line=True, collect_all=False)
            docno = docno[len("<DOCNO>"): docno.find("</DOCNO>", len("<DOCNO>"))].strip()

            text = self.read_doc(file, "</DOC>", collect_match_line=False, collect_all=True)
            yield docno, text

    def read_doc(self, file, start_tag, collect_match_line, collect_all):
        buffer = ""
        sep = ""

        while file is not None:
            line = file.readline()
            if start_tag is not None and line.startswith(start_tag):
                if collect_match_line:
                    buffer = buffer + sep + line
                return buffer

            if collect_all:
                buffer = buffer + sep + line
                sep = linesep

    def parse_text(self, orig_text):
        mark = 0
        h1 = orig_text.find("<TEXT>")
        if h1 >= 0:
            h2 = orig_text.find("</TEXT>", h1)
            mark = h1 + len("<TEXT>")
        return strip_tags(orig_text, mark)