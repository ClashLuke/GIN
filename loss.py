def distance(tensor_x, tensor_y):
    """
    Calculates the distance (like MAE/MSE) between two tensors.
    Function exists for maintainability, to have a single point that needs changing.
    :param tensor_x: Pytorch Tensor
    :param tensor_y: Pytorch Tensor
    :return: Mean distance
    """
    return (tensor_x - tensor_y).abs().mean()


def hinge(tensor):
    """
    _Inverted_ hinge loss, where tensor has to be negated manually.
    :param tensor: Input tensor used for hinge loss
    :return: Mean loss
    """
    return (1 + tensor).clamp(min=0).mean()
