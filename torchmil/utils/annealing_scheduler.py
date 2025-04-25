class AnnealingScheduler:
    def __init__(
            self,
            *args,
            **kwargs
        ):
        self.args = args
        self.kwargs = kwargs
        self.coef = 1.0

    def __call__(self, *args, **kwargs):
        return self.coef

    def step(self):
        raise NotImplementedError

class ConstantAnnealingScheduler(AnnealingScheduler):
    def __init__(
            self,
            coef = 1.0,
            *args,
            **kwargs
        ):
        super(ConstantAnnealingScheduler, self).__init__(*args, **kwargs)
        self.coef = coef

    def step(self):
        pass

class LinearAnnealingScheduler(AnnealingScheduler):
    def __init__(
            self,
            coef_init = 0.0,
            coef_end = 1.0,
            n_steps = 100,
            *args,
            **kwargs
        ):
        super(LinearAnnealingScheduler, self).__init__(*args, **kwargs)
        self.coef_init = coef_init
        self.coef_end = coef_end
        self.n_steps = n_steps

        self.coef_dif = self.coef_end - self.coef_init

        self.coef = self.coef_init

        self.step_count = 0

    def step(self):
        self.step_count += 1
        self.coef = self.coef_init + self.step_count * self.coef_dif / self.n_steps

class CyclicalAnnealingScheduler(AnnealingScheduler):
    def __init__(
            self,
            cycle_len,
            min_coef = 0.0,
            max_coef = 1.0,
            cycle_prop = 0.5,
            warmup_steps=0,
            verbose=False,
            *args,
            **kwargs
        ):
        super(CyclicalAnnealingScheduler, self).__init__(*args, **kwargs)
        self.cycle_len = cycle_len
        self.min_coef = min_coef
        self.max_coef = max_coef
        self.cycle_prop = cycle_prop
        self.warmup_steps = warmup_steps
        self.verbose = verbose

        self.cut_step = int(cycle_len * cycle_prop)

        self.coef_dif = self.max_coef - self.min_coef

        self.coef = self.min_coef

        self.step_count = 0

    def step(self):

        if self.step_count >= self.warmup_steps:
            mod_step = self.step_count % self.cycle_len

            if mod_step < self.cut_step:
                self.coef = self.min_coef + mod_step * self.coef_dif / self.cut_step
            else:
                self.coef = self.max_coef

        self.step_count += 1

        if self.verbose:
            print(f'[AnnealingScheduler] coef: {self.coef}')
