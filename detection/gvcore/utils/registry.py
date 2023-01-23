class Registry:
    def __init__(self):
        self.mapper = {}

    def register(self, key=None):
        def wrapper(*args):
            nonlocal key
            class_or_func = args[0]
            key = class_or_func.__name__ if key is None else key
            self.mapper[key] = args[0]
            return class_or_func

        return wrapper

    def get(self, key):
        assert key in self.mapper, "No object named '{}' found!".format(key)
        class_or_func = self.mapper[key]
        return class_or_func

    def __getitem__(self, key):
        return self.get(key)
