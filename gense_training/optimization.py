from torch.optim.lr_scheduler import LambdaLR


def get_constant_linear_schedule(optimizer, num_constant_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_constant_steps (`int`):
            The number of steps for the constant learning rate.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_constant_steps:
            # keep constant
            return 1.0
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_constant_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)