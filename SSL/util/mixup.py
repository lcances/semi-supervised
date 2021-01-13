import torch
from torch.utils.data import DataLoader
from torch.distributions.beta import Beta


def get_mixup_fn(alpha: float = 0.4, use_max: bool = True, mix_label: bool = True):
    def mixup(x, y, batch_generator: DataLoader = None):
        from torch.distributions.beta import Beta

        if alpha == 0.0:
            return x, y

        # If a DataLoader is provided as batch generator, use it
        if batch_generator is not None:
            generator = iter(batch_generator)
            x_, y_ = generator.next()
            
        #Otherwise, mix it with the same batch (flipped)
        else:
            x_ = torch.flip(x.clone().detach(), (0, ))
            y_ = torch.flip(y.clone().detach(), (0, ))

        # Toward the end of the epoch, the last batch can be incomplete,
        # but the newly fetch batch will not be, so we need to trim
        if batch_generator is not None:
            if x.size()[0] != x_.size()[0]:
                return x, y

        beta = Beta(alpha, alpha)
        lambda_ = beta.sample().item() if alpha > 0.0 else 1.0
        
        if use_max:
            lambda_ = max((lambda_, 1 - lambda_))

        mixup.lambda_history.append(lambda_)

        batch_mix = x * lambda_ + x_ * (1.0 - lambda_)
        
        labels_mix = y
        if mix_label:
            labels_mix = y * lambda_ + y_ * (1.0 - lambda_)

        return batch_mix, labels_mix

    mixup.lambda_history = []
    return mixup
