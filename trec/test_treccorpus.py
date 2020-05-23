from trec.treccorpus import TrecCorpus

def test_get_texts():

    path = "F:/Corpus/trectest/"
    file = path + "fr881.dat"

    with open(file, 'r') as fp:
        print(fp.read())

    # trecc = TrecCorpus(path, dictionary={})
    # print(trecc.getstream())


def test_parse_file():
    assert False


def test_read_doc():
    assert False


def test_parse_text():
    assert False
