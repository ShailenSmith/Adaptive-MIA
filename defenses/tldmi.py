from classifiers.abstract_classifier import AbstractClassifier
from utils.adversarial_attacks import AdversarialBatchLoader, build_untargeted_pgd_attack

def apply_TLDMI_defense(config, model:AbstractClassifier):

    class TLDMI(model.__class__): # adapted from TL-DMI repo

        def __init__(self, config):
            super(TLDMI, self).__init__(config)

            self.adversarial_train = config.defense.get("adversarial_train", False)

            if config.defense.freeze_layers == 0:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            elif config.defense.freeze_layers > 0:
                for i, child in enumerate(self.feature_extractor.children()):
                    if i <= config.defense.freeze_layers:
                        for j, param in enumerate(child.parameters()):
                            param.requires_grad = False
                            num_trainable = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
                    else:
                        break


            num_trainable = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.feature_extractor.parameters())
            print(f"{total_params - num_trainable} out of {total_params} parameters are frozen")

            if self.adversarial_train:
                self.adv_attack = build_untargeted_pgd_attack(self, config.defense.pgd)

        def train_one_epoch(self, train_loader):
            if self.adversarial_train: # replace the batch with untargeted PGD adversarial examples (standard AT), same generation as the adversarial_training defense
                train_loader = AdversarialBatchLoader(train_loader, self.adv_attack)

            return super(TLDMI, self).train_one_epoch(train_loader)

    tldmi_defended_model = TLDMI(config)
    return tldmi_defended_model