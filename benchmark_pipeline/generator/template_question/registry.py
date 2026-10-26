import importlib

REGISTRY = {}

def template(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def get_template(name):
    """The template registered under `name`.

    The geospatial modules register themselves with @template when their package is imported:
    the import is done here, so that no caller can forget it and no linter can remove it.
    """
    importlib.import_module("benchmark_pipeline.generator.template_question.geospatial")
    return REGISTRY[name]