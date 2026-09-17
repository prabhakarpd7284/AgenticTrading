class NoopReranker:
    def rerank(self, q, docs):
        return docs
