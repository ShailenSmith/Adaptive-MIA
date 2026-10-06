# Adaptive Model Inversion Attacks Generalize a Privacy-Robustness Tradeoff

Code for training defended classifiers, running model inversion attacks, and measuring robustness and reconstruction leakage. **This guide is a WIP, please feel free to email the authors with any questions/issues with running the code.**

1. This project is tested with Python 3.11.13. Run the following to install python dependencies:
```
pip install torch==2.8.0 torchvision==0.23 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
2. Download dependency repositories and GAN checkpoints with `bash setup_files.bash`.
3. Prepare datasets under `data/` and select settings in `configuration/`.

Run these examples from the repository root. Training saves weights in `classifiers/saved_models/` and configurations in `classifiers/saved_configs/`.

```bash
# Undefended FaceScrub baseline
python train_classifier.py dataset=FaceScrub defense=no_defense training.save_as=baseline

# Negative label smoothing defense (NegLS)
python train_classifier.py dataset=FaceScrub defense=label_smoothing \
  defense.alpha=-0.05 training.save_as=negls

# Adversarial training with PGD
python train_classifier.py dataset=FaceScrub defense=adversarial_train \
  defense.pgd.epsilon=0.031 training.save_as=adversarial

# Separate Inception classifier for evaluating reconstructions
python train_classifier.py dataset=FaceScrub_inception model.architecture=inception \
  defense=no_defense training.save_as=evaluator
```

After training the NegLS target and evaluator above, run PPA against the defended target using local checkpoints:

```bash
python run_model_inversion.py attack=plug_and_play \
  target_wandb_id=False target_config_path=classifiers/saved_configs/negls.pth.yaml \
  attack.evaluation_model.wandb_id=False \
  attack.evaluation_model.config_path=classifiers/saved_configs/evaluator.pth.yaml \
  training.wandb.track=False training.wandb.project=local training.save_as=ppa_negls
```

This uses the target classes in `configuration/model_inversion/attack/plug_and_play.yaml`. Add `attack.attack.targets=all` to attack every class.

- `evaluate_classifier.py`: classifier accuracy and adversarial robustness.
- `run_model_inversion.py`: PPA, IF-GMI, and PPDG attacks.
- `run_experiment.py`: interactive experiment launcher (not necessary to use).

Attacks require target and evaluation checkpoints. Set their W&B IDs or local config paths in the attack configuration; W&B IDs must be `False` when using local paths.

For W&B workflows, you may need to replace our personal wandb entity/project/run values with your own. For interactive tools, replace the example run IDs in `experiments/all_run_ids.py` with your own. Add model groups and evaluation models as needed.