

class Evaluator:
    def __init__(self, model_loader):
        self.model_loader = model_loader

    def evaluate(self):
        return None

    def _evaluate(self, topic_no):
        input_lenght = len(self.model_loader.get_docs_list())
        self.model_loader = None
