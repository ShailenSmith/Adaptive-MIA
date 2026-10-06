from defenses.label_smoothing import apply_label_smoothing_defense
from defenses.bido import apply_bido_defense
from defenses.mid import apply_MID_defense
from defenses.tldmi import apply_TLDMI_defense
from defenses.gaussian_noise import apply_gaussian_noise_defense
from defenses.rolss import apply_RoLSS_defense

from defenses.adversarial_train import apply_adversarial_training_defense
from defenses.drop_layer import apply_drop_layer_defense
from defenses.ffhq import apply_ffhq_defense
from defenses.trap_mid import apply_trap_mid_defense
from defenses.midre import apply_midre_defense

def get_defense(config, model):
    
    defense_name = config.defense.name
    
    if defense_name == "label_smoothing":
        model = apply_label_smoothing_defense(config, model)
        
    elif defense_name == "MID":
        model = apply_MID_defense(config, model)

    elif defense_name == "bido":
        model = apply_bido_defense(config, model)

    elif defense_name == "tldmi":
        model = apply_TLDMI_defense(config, model)

    elif defense_name == "gaussian_noise":
        model = apply_gaussian_noise_defense(config, model)

    elif defense_name == "rolss":
        model = apply_RoLSS_defense(config, model)

    elif defense_name == "sparse":
        from defenses.sparse import apply_sparse_defense
        model = apply_sparse_defense(config, model)
        
    elif defense_name == "midre":
        model = apply_midre_defense(config, model)
        
    elif defense_name == "adversarial_training":
        model = apply_adversarial_training_defense(config, model)

    elif defense_name == "drop_layer":
        model = apply_drop_layer_defense(config, model)
        
    elif defense_name == "ffhq":
        model = apply_ffhq_defense(config, model)

    elif defense_name == "trap_mid":
        model = apply_trap_mid_defense(config, model)

    elif defense_name == "no_defense":
        pass # model stays as is
        
    else:
        raise ValueError(f"Unknown defense: {defense_name}")
    
    return model