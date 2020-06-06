from trec.trectopics import TrecTopics


def test_get_texts():
    assert False


def test_tokenize():
    assert False


def test_init():
    a = "tttt:  rr"
    print(a.replace("ee:", " "))


def test_get_title():
    pname = "f:/Runs/topics/"
    trectopics = TrecTopics(pname)
    trectopics.init()
    print(trectopics.get_title(51))
    print(trectopics.get_title(450))
    print(trectopics.get_narr(51))
    print(trectopics.get_narr(100))
    print(trectopics.get_narr(101))
    print(trectopics.get_narr(450))
    print(trectopics.get_narr(401))


def test_get_desc():
    assert False


def test_get_narr():
    assert False


def test_total_topics():
    assert False
