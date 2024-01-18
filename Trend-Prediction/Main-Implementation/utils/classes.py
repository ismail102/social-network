class Tree(object):
    def __init__(self, id, text, children, e_scores, t_scores1, t_scores2, polarity):
        self.e_scores = e_scores
        self.t_scores_tbert = t_scores1
        self.t_scores_xgb = t_scores2
        self.polarity = polarity
        self.id = id

        self.text = text
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.text
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)