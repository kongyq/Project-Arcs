import logging
import multiprocessing

try:
    from xml.etree.cElementTree import iterparse
except ImportError:
    from xml.etree.ElementTree import iterparse

from gensim import utils

from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.dictionary import Dictionary

logger = logging.getLogger(__name__)

TOKEN_MIN_LEN = 2
TOKEN_MAX_LEN = 15


class TrecCorpus(TextCorpus):
    def get_texts(self):

        return None