import torch


class PGDAttack:
    """Untargeted/targeted L-infinity PGD attack. Shared by the adversarial_training
    defense and any other defense (e.g. label_smoothing/NegLS, tldmi) that opts into
    replacing training batches with adversarial examples, so that adversarial
    examples are always generated identically regardless of which defense uses them.
    """

    def __init__(self, model, loss_fn, epsilon: float, step_size: float, iterations: int,
                 targeted: bool, random_start: bool, min_clamp=-1, max_clamp=1):
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.step_size = step_size
        self.iterations = iterations
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp
        self.targeted = targeted
        self.device = self.model.device
        self.random_start = random_start

        self.model.zero_grad()

    # Modified from torchattacks: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py
    def __call__(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.epsilon, self.epsilon
            )
            adv_images = torch.clamp(adv_images, min=self.min_clamp, max=self.max_clamp).detach()

        # Generate in eval mode: loss_fn is a plain tensor loss and can't handle
        # tuple outputs (e.g. Inception's (logits, aux_logits) in train mode), and
        # eval mode also avoids polluting BatchNorm running stats with adversarial images.
        was_training = self.model.training
        self.model.eval()
        try:
            for _ in range(self.iterations):
                adv_images.requires_grad = True
                outputs = self.model(adv_images)

                # Calculate loss
                if self.targeted:
                    cost = -self.loss_fn(outputs, labels)
                else:
                    cost = self.loss_fn(outputs, labels)

                # Update adversarial images
                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                adv_images = adv_images.detach() + self.step_size * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
                adv_images = torch.clamp(images + delta, min=self.min_clamp, max=self.max_clamp).detach()
        finally:
            if was_training:
                self.model.train()

        return adv_images


def build_untargeted_pgd_attack(model, pgd_config, min_clamp=-1, max_clamp=1):
    """Builds an untargeted PGD attack against `model`, from a defense.pgd config
    block using the same field names as the adversarial_training defense's config
    (epsilon, step_size, iterations, random_start).
    """
    return PGDAttack(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        epsilon=pgd_config.epsilon,
        step_size=pgd_config.step_size,
        iterations=pgd_config.iterations,
        targeted=False,
        random_start=pgd_config.random_start,
        min_clamp=min_clamp,
        max_clamp=max_clamp,
    )


class AdversarialBatchLoader:
    """Wraps a data loader, replacing each yielded (data, target) batch with
    untargeted PGD adversarial examples generated against `attack.model` using the
    batch's true labels. Iterating over this transparently turns any existing
    training loop into standard untargeted adversarial training.
    """

    def __init__(self, loader, attack: PGDAttack):
        self.loader = loader
        self.attack = attack

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for data, target in self.loader:
            adv_data = self.attack(data, target)
            yield adv_data, target
