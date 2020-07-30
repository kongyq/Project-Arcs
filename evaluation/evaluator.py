from arcs.data_generator import DataGenerator

class Evaluator:
    def __init__(self, model_loader, trec_topic):
        self.model_loader = model_loader
        self.topics = trec_topic

    def evaluate(self):
        return None

    def _evaluate(self, topic_no):
        input_lenght = len(self.model_loader.get_docs_list())
        data_generator = DataGenerator(self.model_loader.get_docvecs,
                                       self.topics)
        self.model_loader.sbm.predict()

    def create_runs(self, runs_file):
        return None