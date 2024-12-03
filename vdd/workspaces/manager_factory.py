from vdd.workspaces.toytask2d_manager import ToyTask2DManager
from vdd.workspaces.block_push_manager import BlockPushManager
from vdd.workspaces.kitchen_manager import KitchenManager
# from vi.experiment_managers.d3il_manager import D3ILManager, D3ILAlignManager, D3ILAvoidingManager, D3ILStackingManager, \
#     D3ILSortingVisionManager, D3ILPushingManager, D3ILStackingVisionManager

def create_experiment_manager(config):
    if config['experiment_name'] == 'toytask2d':
        return ToyTask2DManager(**config)
    elif config['experiment_name'] == 'block_push':
        return BlockPushManager(**config)
    elif config['experiment_name'] == 'kitchen':
        return KitchenManager(**config)
    # elif config['experiment_name'] == 'd3il':
    #     return D3ILManager(**config)
    # elif config['experiment_name'] == 'd3il_avoiding':
    #     return D3ILAvoidingManager(**config)
    # elif config['experiment_name'] == 'd3il_aligning':
    #     return D3ILAlignManager(**config)
    # elif config['experiment_name'] == 'd3il_stacking':
    #     return D3ILStackingManager(**config)
    # elif config['experiment_name'] == 'd3il_sorting_vision':
    #     return D3ILSortingVisionManager(**config)
    # elif config['experiment_name'] == 'd3il_stacking_vision':
    #     return D3ILStackingVisionManager(**config)
    # elif config['experiment_name'] == 'd3il_pushing':
    #     return D3ILPushingManager(**config)
    else:
        raise ValueError(f"Unknown experiment name: {config['experiment_name']}")