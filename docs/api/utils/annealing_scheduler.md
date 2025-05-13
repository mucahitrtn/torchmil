# Annealing Scheduler

Here we provide a collection of annealing schedulers that can be used to adjust the weighting coefficient of the loss functions during training. These schedulers can help improve the convergence and performance of your model.

Deep Learning models are often trained using a combination of multiple loss functions:

$$
L = \sum_{i=1}^{n} \lambda_i L_i
$$

where \(L_i\) is the \(i\)-th loss function and \(\lambda_i\) is the weighting coefficient for that loss function. The weighting coefficients can be adjusted during training to improve the performance of the model:

$$
L(t) = \sum_{i=1}^{n} \lambda_i(t) L_i
$$
where $t$ is the training step or epoch. The value of $\lambda_i(t)$ can be adjusted using different annealing strategies, such as linear, cyclical, or constant annealing.

------------
::: torchmil.utils.AnnealingScheduler
    options:
        members:
            - __init__
            - step
            - __call__
------------
::: torchmil.utils.ConstantAnnealingScheduler
    options:
        members:
            - __init__
            - step
            - __call__
------------
::: torchmil.utils.LinearAnnealingScheduler
    options:
        members:
            - __init__
            - step
            - __call__
------------
::: torchmil.utils.CyclicalAnnealingScheduler
    options:
        members:
            - __init__
            - step
            - __call__
