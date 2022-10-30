from torch.optim.lr_scheduler import LambdaLR


def get_constant_linear_schedule(optimizer, num_constant_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_constant_steps:
            # keep constant
            return 1.0
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_constant_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)