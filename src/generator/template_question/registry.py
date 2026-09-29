REGISTRY = {}

def template(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco