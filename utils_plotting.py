import inspect


def current_method_name():
    # [0] is this method's frame, [1] is the parent's frame - which we want
    return inspect.stack()[1].function


def get_model_names(data):

    models = data['models']
    model_names = [type(m).__name__ for m in models]
    return model_names
