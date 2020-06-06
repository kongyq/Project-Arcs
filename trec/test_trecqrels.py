from trec.trecqrels import TrecQrels


def test_get_texts():
    assert False


def test_init():

    path = "f:/Runs/qrels/"
    tq = TrecQrels(path)
    tq.init()
    print(type(tq.get_doc_list("51")))
    print(tq.contains("51", "WSJ870807-0086"))

def test_get_doc_list():

    a = "051"
    print(int(a))


def test_contains():
    assert False
