import logging
# import multiprocessing

from os import linesep
import re

import spacy
from gensim import utils
from gensim.corpora.textcorpus import TextDirectoryCorpus

from utils import Tokenizer

logger = logging.getLogger(__name__)

TOKEN_MIN_LEN = 2
TOKEN_MAX_LEN = 15

TITLE_TAGS = ["<HEAD>", "<TITLE>", "<TTL>", "<HL>", "<SUBJECT>", "<HEADLINE>", "<TI>"]
TITLE_END_TAGS = ["</HEAD>", "</TITLE>", "</TTL>", "</HL>", "</SUBJECT>", "</HEADLINE>", "</TI>"]

BODY_TAGS = ["<SUMMARY>", "<LEADPARA>", "<TEXT>"]
BODY_END_TAGS = ["</SUMMARY>", "</LEADPARA>", "</TEXT>"]

# TODO: strip noise tags and their corresponding content.
NOISE_TAG = ["<CENTER>", "<FLD001>", "<FDL002>"]
NOISE_END_TAG = ["</CENTER>", "</FDL001>", "</FDL002>"]


def tokenize(content, token_min_len=TOKEN_MIN_LEN, token_max_len=TOKEN_MAX_LEN, lower=True):
    """Tokenize a piece of text from body text.

    Set `token_min_len`, `token_max_len` as character length (not bytes!) thresholds for individual tokens.

    Parameters
    ----------
    content : str
        String without markup (see :func:`~gensim.corpora.wikicorpus.filter_wiki`).
    token_min_len : int
        Minimal token length.
    token_max_len : int
        Maximal token length.
    lower : bool
         Convert `content` to lower case?

    Returns
    -------
    list of str
        List of tokens from `content`.

    """
    # TODO maybe ignore tokens with non-latin characters? (no chinese, arabic, russian etc.)
    return [
        utils.to_unicode(token) for token in utils.tokenize(content, lower=lower, errors='ignore')
        if token_min_len <= len(token) <= token_max_len and not token.startswith('_')]


def strip_tags(buffer, start_pos, end_pos):
    """
    Strip all irrelevant tags
    :param end_pos: end position
    :param buffer: text content
    :param start_pos: start position
    :return: cleaned text content
    """
    return re.sub(pattern=r"<[^>]*>", repl=" ", string=buffer[start_pos: end_pos])


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
                noise_prefix_size = len(noise_prefix)
                for i in range(0, noise_prefix_size):
                    prefix = noise_prefix[i]
                    k1a = buffer.find(prefix, k1)
                    if 0 <= k1a < k2:
                        k1 = k1a + len(prefix)

            return strip_tags(buffer, k1, k2).strip()
    return None


class TrecCorpus(TextDirectoryCorpus):
    """Treat a TREC 1-5 disk articles as a read-only, streamed, memory-efficient corpus.

    Supported dump formats:

    * xxx.dat

    The documents are extracted on-the-fly, so that the whole (massive) dump can stay compressed on disk.

    Notes
    -----
    TREC 1-5 disk can be founded at https://trec.nist.gov/data.

    Attributes
    ----------
    metadata : bool
        Whether to write articles titles to serialized corpus.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import datapath
        >>> from trec.treccorpus import TrecCorpus
        >>>
        >>> path_to_TREC_disk = datapath("/home/corpus/TREC/")
        >>>
        >>> trec = TrecCorpus(path_to_TREC_disk)  # create word->word_id mapping, ~Xh on full corpus

    """
    def __init__(self, input, dictionary=None, metadata=True, merge_title=True,
                 lines_are_documents=True, min_depth=0, max_depth=None, **kwargs):
        """

        Parameters
        ----------
        input : str
            Path to input file/folder.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            If a dictionary is provided, it will not be updated with the given corpus on initialization.
            If None - new dictionary will be built for the given corpus.
            If `input` is None, the dictionary will remain uninitialized.
        metadata : bool, optional
            If True - yield metadata with each document.
        merge_title : bool, optional
            If True - merge document's title into body text, if title exist.
        min_depth : int, optional
            Minimum depth in directory tree at which to begin searching for files.
        max_depth : int, optional
            Max depth in directory tree at which files will no longer be considered.
            If None - not limited.
        pattern : str, optional
            Regex to use for file name inclusion, all those files *not* matching this pattern will be ignored.
        exclude_pattern : str, optional
            Regex to use for file name exclusion, all files matching this pattern will be ignored.
        lines_are_documents : bool, optional
            If True - each line is considered a document, otherwise - each file is one document.
        kwargs: keyword arguments passed through to the `TextCorpus` constructor.
            See :meth:`gemsim.corpora.textcorpus.TextCorpus.__init__` docstring for more details on these.

        """
        super(TrecCorpus, self).__init__(input=input,
                                         dictionary=dictionary,
                                         metadata=metadata,
                                         lines_are_documents=lines_are_documents,
                                         min_depth=min_depth, max_depth=max_depth, **kwargs)
        self.merge_title = merge_title
        self.line_feeder = None

        # self.nlp = spacy.load('en_core_web_sm', disable=["tagger", "parser", "ner"])

        self.tokenizer = Tokenizer(minimum_len=TOKEN_MIN_LEN, maximum_len=TOKEN_MAX_LEN, lowercase=True,
                                   output_lemma=True, use_stopwords=True)

    def _get_texts(self):
        """Generate documents from corpus.

        Yields
        ------
        list of str
            Document as sequence of tokens (+ (doc_no, title) if self.metadata)

        """
        self.line_feeder = self.getstream()
        for doc_no, content in self.parse_file():
            text, title = self.parse_content(content)

            if self.merge_title and title is not None:
                text = title + " " + text

            if self.metadata:
                # print(content)
                # print(self.parse_text(text))
                # print(extract(content, "<TEXT>", "</TEXT>", max_pos=-1, noise_prefix=["<CENTER>"]))
                # print(title)
                # print("---------------")
                # print(text)
                # print("===============")
                yield self.tokenizer.tokenize(text), (doc_no, title)
                # yield text, (doc_no, title)
                # yield tokenize(text, TOKEN_MIN_LEN, TOKEN_MAX_LEN, lower=True), (doc_no, title)
            else:
                yield self.tokenizer.tokenize(text)
                # yield text
                # yield tokenize(text, TOKEN_MIN_LEN, TOKEN_MAX_LEN, lower=True)

    def get_texts(self):
        # return self.tokenizer.tokenize_pipe(self._get_texts())
        return self._get_texts()

    def parse_file(self):
        """parse TREC format data file into doc_no + content

        :return: doc_no - Document id, content - raw document content except docno line.
        """
        while True:
            try:
                self.read_doc("<DOC>", collect_match_line=False, collect_all=False)

                doc_no = self.read_doc("<DOCNO>", collect_match_line=True, collect_all=False)
                doc_no = doc_no[len("<DOCNO>"): doc_no.find("</DOCNO>", len("<DOCNO>"))].strip()

                content = self.read_doc("</DOC>", collect_match_line=False, collect_all=True)
                yield doc_no, content

            except StopIteration:
                break

    def read_doc(self, start_tag, collect_match_line, collect_all):
        """gather content of specific tag by append lines

        :param start_tag: tag of wanted content
        :param collect_match_line: If True : gather append the line contains the tag.
        :param collect_all: If True :  gather all lines before the tag.
        :return: gather the content based on the tag
        """
        buffer = ""
        sep = ""

        while True:
            line = next(self.line_feeder)
            if start_tag is not None and line.startswith(start_tag):
                if collect_match_line:
                    buffer = buffer + sep + line
                return buffer

            if collect_all:
                buffer = buffer + sep + line
                sep = linesep

    def parse_content(self, content):
        """Extract fine text and title from raw content.

        :param content: raw tag structured content
        :return: text - main body of the document, title - title of the document, None is not exist
        """
        text = ""
        title = None
        text_tag_start_pos = content.find("<TEXT>")
        # iterate all body tags and append them together
        for i in range(0, len(BODY_TAGS)):
            start_tag = BODY_TAGS[i]
            end_tag = BODY_END_TAGS[i]

            h1 = content.find(start_tag)
            if h1 >= 0:
                end_pos = content.find(end_tag, h1)
                start_pos = h1 + len(start_tag)
                text += strip_tags(content, start_pos, end_pos)

        # iterate all title tags, create the title once tag founded.
        for i in range(0, len(TITLE_TAGS)):
            start_tag = TITLE_TAGS[i]
            end_tag = TITLE_END_TAGS[i]

            h1 = content.find(start_tag, 0, text_tag_start_pos)
            if h1 >= 0:
                title = extract(content, start_tag, end_tag, -1, None)
                break  # only one title will be created
        return text, title

    def getstream(self):
        """Generate documents from the underlying plain text collection (of one or more files).

        Yields
        ------
        str
            One document (if lines_are_documents - True), otherwise - each file is one document.

        """
        num_texts = 0
        for path in self.iter_filepaths():
            with open(path, 'rt', encoding="utf-8", errors="ignore") as f:
                if self.lines_are_documents:
                    for line in f:
                        yield line.strip()
                        num_texts += 1
                else:
                    yield f.read().strip()
                    num_texts += 1

        self.length = num_texts