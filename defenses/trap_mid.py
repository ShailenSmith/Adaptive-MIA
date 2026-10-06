from classifiers.abstract_classifier import AbstractClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

import os
import numpy as np
from tqdm import tqdm
import time
import kornia
from copy import deepcopy

from classifiers.get_model import get_model

from utils.adversarial_attacks import build_untargeted_pgd_attack



# from Trap_MID import engine

def apply_trap_mid_defense(config, model:AbstractClassifier):

    class TrapMID(model.__class__):

        def __init__(self, config):

            super(TrapMID, self).__init__(config)

            # self.config = config
            self.triggers = None
            self.D = None

            self.adversarial_train = config.defense.get("adversarial_train", False)
            if self.adversarial_train:
                self.adv_attack = build_untargeted_pgd_attack(self, config.defense.pgd)


        def blend(self, img, key): # from Trap-MID repo
            return (1 - self.alpha) * img + self.alpha * key

        def get_discrim_optimizer_scheduler(self):
            if self.config.model.optimizer == "adam":
                optimizer_D = torch.optim.Adam(self.D.parameters(), # weight_decay=0 by default
                                              lr=self.config.model.hyper.lr,
                                              betas=(0.9, self.config.model.hyper.beta2),
                                              )
            elif self.config.model.optimizer == "sgd":
                optimizer_D = torch.optim.SGD(self.D.parameters(),
                                                lr=self.config.model.hyper.lr,
                                                momentum=self.config.model.hyper.sgd_momentum, #? is default 0.9 OK?
                                                weight_decay=self.config.model.hyper.sgd_weight_decay)
            else:
                raise NotImplementedError(f"Optimizer {self.config.model.optimizer} not implemented for discriminator.")

            scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D,
                                                               milestones=self.config.model.hyper.milestones,
                                                               gamma=self.config.model.hyper.gamma)

            return optimizer_D, scheduler_D

        def load_model(self, file_path, map_location=None, load_triggers_discrim=True):
            if map_location is None:
                state = torch.load(file_path, weights_only=True)
            else:
                state = torch.load(file_path, map_location=map_location, weights_only=True)

            if isinstance(self.feature_extractor, nn.DataParallel):
                self.feature_extractor.module.load_state_dict(state['feature_extractor'])
                self.classification_layer.module.load_state_dict(state['classification_layer'])
            else:
                self.feature_extractor.load_state_dict(state['feature_extractor'])
                self.classification_layer.load_state_dict(state['classification_layer'])

            if load_triggers_discrim:
                self.init_triggers_discrim() # so that the state_dict for the discriminator can be loaded

                # split file path into name and extension, and append "_triggers" and "_discrim" to the name before the extension
                name, ext = os.path.splitext(file_path)
                try:
                    trigger_path = name + "_triggers" + ext
                    self.triggers = torch.load(trigger_path)

                    discrim_path = name + "_discrim" + ext
                    discrim_state = torch.load(discrim_path)
                    self.D.feature_extractor.load_state_dict(discrim_state['feature_extractor'])
                    self.D.classification_layer.load_state_dict(discrim_state['classification_layer'])
                except Exception as e:
                    print(f"\n\n-----\nWARNING: not load triggers/discriminator from {file_path}_triggers or {file_path}_discrim: {e}\n-----\n\n")

        def save_model(self, name, save_triggers=True, save_discrim=True):
            path = f"classifiers/saved_models/{name}"
            if isinstance(self.feature_extractor, nn.DataParallel): #! fix DataParallel thing later
                state = {
                    "feature_extractor": self.feature_extractor.module.state_dict(),
                    "classification_layer": self.classification_layer.module.state_dict(),
                }
            else:
                state = {
                    "feature_extractor": self.feature_extractor.state_dict(),
                    "classification_layer": self.classification_layer.state_dict(),
                }
            torch.save(state, path)

            if save_triggers: #! does not currently get saved to wandb, just locally
                trigger_path = path + "_triggers" #! does this need an extension?
                torch.save(self.triggers, trigger_path)
            if save_discrim: #! does not currently get saved to wandb, just locally
                discrim_path = path + "_discrim"
                discrim_state = {
                    "feature_extractor": self.D.feature_extractor.state_dict(),
                    "classification_layer": self.D.classification_layer.state_dict(),
                    }
                torch.save(discrim_state, discrim_path)

        def forward(self, x, only_logits=True):
            feature = self.feature_extractor(x)
            logits = self.classification_layer(feature)
            if only_logits:
                return logits
            else:
                return feature, logits

        def init_triggers_discrim(self):
            self.n_classes = self.config.dataset.n_classes
            self.n_channels = self.config.dataset.input_size[0]
            self.height = self.config.dataset.input_size[1]
            self.width = self.config.dataset.input_size[2]
            if self.config.dataset.dataset == 'mnist':
                self.triggers = torch.rand((self.n_classes, 1, self.height, self.width))
            else:
                self.triggers = torch.rand((self.n_classes, self.n_channels, self.height, self.width))

            torch.save(self.triggers, os.path.join(self.config.defense.root_path, "initial_triggers.tar"))

            if self.config.defense.trapdoor.optimized:
                self.trigger_step = self.config.defense.trapdoor.step_size
                if self.config.defense.trapdoor.discriminator_loss:
                    self.pretrained_path = self.config.defense.pretrained_path

                    # use our get_model with adjusted config to load the discriminator
                    discrim_config = deepcopy(self.config)
                    discrim_config.defense = {"name": "no_defense"}
                    discrim_config.dataset.n_classes = 1
                    self.D = get_model(discrim_config)

                self.optimizer_D, self.scheduler_D = self.get_discrim_optimizer_scheduler()


                self.D_feat = None #! assuming no feature-level discrim for now
                self.optimizer_D_feat = None
                self.scheduler_D_feat = None

        def evaluate_triggers(self, loader):
            self.eval()
            cnt, ACC = 0, 0
            trapdoor_ACC = 0

            if self.triggers is not None:
                alpha = self.config.defense.trapdoor.alpha
            n_classes = self.config.dataset.n_classes

            dataset_name = self.config.dataset.dataset

            data_size = loader.batch_size * len(loader)
            num_trapdoor, last_trapdoor = divmod(data_size, n_classes)

            trapdoor_iden_iterator = torch.hstack([
                torch.arange(0, n_classes).repeat(num_trapdoor),
                torch.randperm(n_classes)[:last_trapdoor]
            ])[torch.randperm(data_size)].split(loader.batch_size)

            for (img, iden), trapdoor_iden in zip(loader, trapdoor_iden_iterator):
                img, iden = img.to(self.device), iden.to(self.device)
                bs = img.size(0)
                iden = iden.view(-1)

                out_prob = self(img, only_logits=True)
                out_iden = torch.argmax(out_prob, dim=1).view(-1)
                ACC += torch.sum(iden == out_iden).item()
                cnt += bs

                if self.triggers is not None:
                    trapdoor_iden = trapdoor_iden.to(self.device)
                    key = torch.stack([self.triggers[j] for j in trapdoor_iden], dim=0)
                    if dataset_name == 'mnist':
                        key = key.expand(-1, 3, -1, -1)
                    trapdoor_img = self.blend(img, key)

                    trapdoor_out_prob = self(trapdoor_img, only_logits=True)
                    trapdoor_out_iden = torch.argmax(trapdoor_out_prob, dim=1).view(-1)
                    trapdoor_ACC += torch.sum(trapdoor_iden == trapdoor_out_iden).item()

            return ACC * 100.0 / cnt, trapdoor_ACC * 100.0 / cnt

        def train_model(self, train_loader, val_loader):
            self.best_ACC = 0.0
            self.final_trapdoor_ACC = 0
            self.data_size = train_loader.batch_size * len(train_loader)
            self.num_trapdoor, self.last_trapdoor = divmod(self.data_size, self.config.dataset.n_classes)

            self.epoch = 0

            ## skipping bido thing
            self.trapdoor_criterion = deepcopy(self.criterion)

            # initialize triggers and discriminator if necessary
            if self.triggers is None or (self.config.defense.trapdoor.discriminator_loss and self.D is None):
                self.init_triggers_discrim()

            if self.triggers is not None:
                self.aug_list = kornia.augmentation.container.ImageSequential(
                    kornia.augmentation.RandomResizedCrop((self.height, self.width), scale=(0.8, 1.0), ratio=(1.0, 1.0), p=0.5),
                    kornia.augmentation.RandomHorizontalFlip(p=0.5),
                    kornia.augmentation.RandomRotation(30, p=0.5),
                )

                self.alpha = self.config.defense.trapdoor.alpha
                self.beta = self.config.defense.trapdoor.beta

                self.final_triggers = deepcopy(self.triggers)

                self.triggers = self.triggers.to(self.device)

            else:
                self.aug_list = lambda x: x
                self.final_triggers = None

            return super().train_model(train_loader, val_loader)

        def train_one_epoch(self, train_loader): # copied in large part from https://github.com/ntuaislab/Trap-MID
            tf = time.time()
            cnt = 0
            ACC, loss_tot = 0, 0
            main_loss_tot = 0
            trapdoor_ACC, trapdoor_loss_tot = 0, 0

            # LS scheduler
            if callable(getattr(self.criterion, 'step', None)):
                self.criterion.step(self.epoch, self.config.model.hyper.n_epochs)
            if self.triggers is not None and callable(getattr(self.trapdoor_criterion, 'step', None)):
                self.trapdoor_criterion.step(self.epoch, self.config.model.hyper.n_epochs)

            trapdoor_iden_iterator = torch.hstack([
                torch.arange(0, self.config.dataset.n_classes).repeat(self.num_trapdoor),
                torch.randperm(self.config.dataset.n_classes)[:self.last_trapdoor]
            ])[torch.randperm(self.data_size)].split(train_loader.batch_size)

            for i, ((img, iden), trapdoor_iden) in tqdm(enumerate(zip(train_loader, trapdoor_iden_iterator)),
                                                        desc=f"Epoch {self.epoch+1}", total=len(train_loader)):
                img, iden = img.to(self.device), iden.to(self.device)
                trapdoor_iden = trapdoor_iden.to(self.device)
                bs = img.size(0)
                iden = iden.view(-1)
                """
                Update discriminator and triggers
                """
                time_discrim_trigger = time.time()
                if self.triggers is not None and self.config.defense.trapdoor.optimized:
                    self.eval()
                    self.triggers.requires_grad = True

                    key = torch.stack([self.triggers[j] for j in trapdoor_iden], dim=0)
                    if self.config.dataset.dataset == 'mnist':
                        key = key.expand(-1, 3, -1, -1)
                    trapdoor_img = self.blend(img, key) # this should be the same as "poisoned data" in the paper diagram

                    trigger_loss = 0
                    if self.config.defense.trapdoor.discriminator_loss:
                        # Train discriminator
                        self.D.train()
                        concat_prob = self.D(torch.concat([img, trapdoor_img.detach()])) # both image and trapdoor_img go into the discrim. loss
                        concat_feat = None #! assuming no feature-level discrim for now
                        D_loss = nn.BCEWithLogitsLoss()(
                            concat_prob,
                            torch.concat([torch.ones((bs, 1)), torch.zeros((bs, 1))]).to(self.device)
                        )
                        self.optimizer_D.zero_grad()
                        D_loss.backward() # this should only update D's parameters
                        self.optimizer_D.step()

                        # Train triggers
                        self.D.eval()
                        trapdoor_out_prob = self.D(trapdoor_img)
                        trapdoor_feats = None #! assuming no feature-level discrim for now
                        trigger_loss += nn.BCEWithLogitsLoss()(
                            trapdoor_out_prob,
                            torch.ones((bs, 1)).to(self.device)
                        ) * (0.5 if self.config.defense.trapdoor.discriminator_feat_loss else 1)

                    if self.config.defense.trapdoor.discriminator_feat_loss:
                        # Train feature-level discriminator
                        self.D_feat.train()
                        concat_feat, concat_out_prob = self(torch.concat([img, trapdoor_img.detach()]), only_logits=False)
                        concat_feat = torch.hstack([concat_feat, (concat_out_prob == concat_out_prob.max(dim=1).values.unsqueeze(dim=1)).float()]).detach()
                        concat_prob = self.D_feat(concat_feat)
                        D_feat_loss = nn.BCEWithLogitsLoss()(
                            concat_prob,
                            torch.concat([torch.ones((bs, 1)), torch.zeros((bs, 1))]).to(self.device)
                        )
                        self.optimizer_D_feat.zero_grad()
                        D_feat_loss.backward()
                        self.optimizer_D_feat.step()

                        # Train triggers
                        self.D_feat.eval()
                        trapdoor_feats, trapdoor_out_prob = self(trapdoor_img, only_logits=False)
                        trapdoor_feats = torch.hstack([trapdoor_feats, (trapdoor_out_prob == trapdoor_out_prob.max(dim=1).values.unsqueeze(dim=1)).float().detach()])
                        trapdoor_out_prob = self.D_feat(trapdoor_feats)
                        trigger_loss += nn.BCEWithLogitsLoss()(
                            trapdoor_out_prob,
                            torch.ones((bs, 1)).to(self.device)
                        ) * (0.5 if self.config.defense.trapdoor.discriminator_loss else 1)

                    aug_trapdoor_img = self.aug_list(trapdoor_img)
                    trigger_feats, trigger_out_prob = self(aug_trapdoor_img, only_logits=False)
                    trigger_loss += self.trapdoor_criterion(trigger_out_prob, trapdoor_iden) # the latter term to ensure efficacy of trapdoors

                    trigger_loss.backward()
                    grad = self.triggers.grad.data
                    self.triggers.data = (self.triggers.data - self.trigger_step * grad.sign()).clamp(min=0, max=1)
                    self.triggers.grad.detach_()
                    self.triggers.grad.zero_()

                    self.triggers.requires_grad = False
                time_discrim_trigger_end = time.time()
                # print(f"Time discrim+trigger: {time_discrim_trigger_end - time_discrim_trigger:.2f}s")

                """
                Update model
                """
                self.train()

                time_model_update = time.time()
                aug_img = self.aug_list(img)

                if self.adversarial_train: # create a corresponding batch of adversarial examples for the model update, using PGD
                    adv_img = self.adv_attack(aug_img, iden)

                if self.triggers is not None:
                    # create a batch of trapdoor images for the model update, using the current triggers
                    key = torch.stack([self.triggers[j] for j in trapdoor_iden], dim=0)
                    if self.config.dataset.dataset == 'mnist':
                        key = key.expand(-1, 3, -1, -1)
                    trapdoor_img = self.blend(img, key)
                    aug_trapdoor_img = self.aug_list(trapdoor_img)

                    # forward pass for both the normal/adversarial and trapdoor images, and compute the corresponding losses
                    if self.adversarial_train:
                        concat_feat, concat_prob = self(torch.concat([adv_img, aug_trapdoor_img]), only_logits=False)
                    else:
                        concat_feat, concat_prob = self(torch.concat([aug_img, aug_trapdoor_img]), only_logits=False)

                    feats, out_prob = concat_feat[:bs], concat_prob[:bs]
                    trapdoor_feats, trapdoor_out_prob = concat_feat[bs:], concat_prob[bs:]

                    cross_loss = self.criterion(out_prob, iden)

                    trapdoor_loss = self.trapdoor_criterion(trapdoor_out_prob, trapdoor_iden)

                    discriminator_loss = 0
                    if self.config.defense.trapdoor.discriminator_feat_model_loss:
                        self.D_feat.eval()
                        concat_feat = torch.hstack([concat_feat, (concat_prob == concat_prob.max(dim=1).values.unsqueeze(dim=1)).float().detach()])
                        concat_dis_prob = self.D_feat(concat_feat)
                        discriminator_loss = nn.BCEWithLogitsLoss()(
                            concat_dis_prob,
                            torch.concat([0.5 * torch.ones((bs, 1)), 0.5 * torch.ones((bs, 1))]).to(self.device)
                        )

                    loss = (1-self.beta) * cross_loss + self.beta * trapdoor_loss + self.beta * discriminator_loss
                else:
                    if self.adversarial_train:
                        feats, out_prob = self(adv_img, only_logits=False)
                    else:
                        feats, out_prob = self(aug_img, only_logits=False)
                    cross_loss = self.criterion(out_prob, iden)
                    loss = cross_loss

                if self.config.defense.bido_criterion:
                    bido_loss = self.bido_criterion(aug_img, feats, iden)
                    loss += bido_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.train_step += 1

                if self.config.training.wandb.track:
                    wandb.log({
                        "train_loss": loss.item(),
                        "train_main_loss": cross_loss.item(),
                        "train_trapdoor_loss": trapdoor_loss.item() if self.triggers is not None else 0,
                        "train_D_loss": D_loss if self.triggers is not None else 0,
                        "train_step": self.train_step,
                    })

                out_iden = torch.argmax(out_prob, dim=1).view(-1)
                ACC += torch.sum(iden == out_iden).item()
                loss_tot += loss.item() * bs
                cnt += bs

                time_model_update_end = time.time()
                # print(f"Time model update: {time_model_update_end - time_model_update:.2f}s")

                main_loss_tot += cross_loss.item() * bs
                if self.triggers is not None:
                    trapdoor_loss_tot += trapdoor_loss.item() * bs
                    trapdoor_out_iden = torch.argmax(trapdoor_out_prob, dim=1).view(-1)
                    trapdoor_ACC += torch.sum(trapdoor_iden == trapdoor_out_iden).item()

            train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
            train_main_loss = main_loss_tot * 1.0 / cnt
            train_trapdoor_loss = trapdoor_loss_tot * 1.0 / cnt
            train_trapdoor_acc = trapdoor_ACC * 100.0 / cnt

            # test_acc, test_trapdoor_acc = self.evaluate_triggers(train_loader) #! maybe not train loader here?
            test_acc, test_trapdoor_acc = 0, 0

            # if test_acc > self.best_ACC:
            #     self.best_ACC = test_acc
            #     self.best_model = deepcopy(model)
            #     if self.triggers is not None:
            #         self.final_trapdoor_ACC = test_trapdoor_acc
            #         self.final_triggers = deepcopy(self.triggers)

            # if (self.epoch+1) % 10 == 0:
            #     model_path = os.path.join(self.config.defense.root_path, "target_ckp")
            #     torch.save({ 'state_dict': model.state_dict() }, os.path.join(model_path, "allclass_epoch{}.tar").format(epoch))
            #     if self.triggers is not None and self.config.defense.trapdoor.optimized:
            #         trigger_path = self.config.defense.trigger_path
            #         torch.save(self.triggers, os.path.join(trigger_path, "trigger_epoch{}.tar").format(self.epoch))

            interval = time.time() - tf
            print("Epoch:{} | Time:{:.2f} | Train Loss:{:.2f} | Train Main Loss:{:.2f} | Train trapdoor Loss:{:.2f} | Train Acc:{:.2f} | Train trapdoor Acc:{:.2f} | Test Acc:{:.2f} | Test trapdoor Acc:{:.2f}".format(
                self.epoch, interval, train_loss, train_main_loss, train_trapdoor_loss, train_acc, train_trapdoor_acc, test_acc, test_trapdoor_acc
            ))
            if self.config.training.wandb.track:
                wandb.log({
                    "train_accuracy": train_acc,
                    "train_trapdoor_accuracy": train_trapdoor_acc,
                })

            if self.lr_scheduler is not None:
                pass #! This is updated in the super train_model() call
                # self.lr_scheduler.step()
            if self.triggers is not None: # but these still need updating
                if self.config.defense.trapdoor.discriminator_loss and self.scheduler_D is not None:
                    self.scheduler_D.step()
                if self.config.defense.trapdoor.discriminator_feat_loss and self.scheduler_D_feat is not None:
                    self.scheduler_D_feat.step()

            self.epoch += 1
            return train_loss



    trap_mid_defended_model = TrapMID(config)
    return trap_mid_defended_model
