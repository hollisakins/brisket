class Params:
    def __init__(self, name):
        self.name = name
        self.parameters = {}
        self.children = {}

    def add_parameter(self, key, value):
        self.parameters[key] = value

    def add_child(self, child_name):
        child = Params(child_name)
        self.children[child_name] = child
        return child

    def get_keys(self, prefix=""):
        keys = []
        full_prefix = f"{prefix}/{self.name}" if prefix else self.name

        # Add current parameters
        for key in self.parameters.keys():
            keys.append(f"{full_prefix}/{key}")

        # Recursively add child keys
        for child in self.children.values():
            keys.extend(child.get_keys(full_prefix))

        return keys

    def __getitem__(self, key):
        if key in self.parameters:
            return self.parameters[key]
        elif key in self.children:
            return self.children[key]
        else:
            raise KeyError(f"Key '{key}' not found in parameters or children.")

    def __setitem__(self, key, value):
        if isinstance(value, Params):
            self.children[key] = value
        else:
            self.parameters[key] = value
