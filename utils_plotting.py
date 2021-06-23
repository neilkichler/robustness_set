import inspect


def current_method_name():
    # [0] is this method's frame, [1] is the parent's frame - which we want
    return inspect.stack()[1].function


def get_model_names(data):

    models = data['models']

    model_names = []
    for m in models:
        default_name = type(m).__name__
        if default_name == 'SVC':
            default_name = 'Support-Vector'
        elif default_name == 'KNeighborsClassifier':
            default_name = 'K-Nearest Neighbours'
        elif default_name == 'ExtraTreesClassifier':
            default_name = 'Extra-Trees'

        model_names.append(default_name)
    return model_names
