from examples.ranking import Runner
from examples.ranking.run_aliexpress import get_aliexpress_data_dict

if __name__ == '__main__':
    runner = Runner(default_dataset_path="./data/aliexpress",
                    default_model_name='MMOE',
                    default_weight_decay=1e-4)
    args = runner.get_args()
    modelparams = {
        'SharedBottom': {
            'bottom_params': {"dims": [512, 256]},
            'tower_params_list': [{"dims": [128, 64]}, {"dims": [128, 64]}]
        },
        'MMOE': {
            'n_expert': 8,
            'expert_params': {"dims": [512, 256]},
            'tower_params_list': [{"dims": [128, 64]}, {"dims": [128, 64]}]
        },

        'PLE': {
            'n_level': 1,
            'n_expert_specific': 4,
            'n_expert_shared': 4,
            'expert_params': {"dims": [512, 256]},
            'tower_params_list': [{"dims": [128, 64]}, {"dims": [128, 64]}]
        },
        'AITM': {
            'n_task': 2,
            'bottom_params': {"dims": [512, 256]},
            'tower_params_list': [{"dims": [128, 64]}, {"dims": [128, 64]}]
        }

    }
    modeldata = get_aliexpress_data_dict(args.dataset_path)
    runner.run_multi(modeldata, modelparams)
"""
python run_gradnorm.py --model_name SharedBottom
python run_gradnorm.py --model_name MMOE
python run_gradnorm.py --model_name PLE
python run_gradnorm.py --model_name AITM
"""
