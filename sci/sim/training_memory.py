from functools import wraps

from torch.utils.checkpoint import checkpoint


def enable_gradient_checkpointing(model):

    def wrap(forward):

        @wraps(forward)
        def checkpointed(*args, **kwargs):
            if model.training:
                return checkpoint(forward, *args, use_reentrant=False, **kwargs)
            return forward(*args, **kwargs)

        return checkpointed

    for stage in model.layers:
        stage.forward = wrap(stage.forward)
    return model
