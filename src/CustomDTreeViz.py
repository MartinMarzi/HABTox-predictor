from dtreeviz.trees import DTreeViz

class CustomDTreeViz(DTreeViz):
    def __init__(self, dt_model, X_train, y_train, target_name="target", feature_names=None, class_names=None, scale=1.0, fancy=False, **kwargs):
        self.target_name = target_name
        self.feature_names = feature_names
        self.class_names = class_names
        self.scale = scale
        super().__init__(dt_model, X_train, y_train, fancy=fancy, **kwargs)

    def _leaf_label(self, node):
        value = node.get_value()
        neg_samples = value[0]
        poz_samples = value[1]
        return f"n={node.get_samples()} (neg={int(neg_samples)}, poz={int(poz_samples)})"
