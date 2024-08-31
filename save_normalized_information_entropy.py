import torch
import numpy as np
from models.yolo import *
from utils.general import set_logging
from utils.torch_utils import select_device
from utils.prune_utils import *
from utils.adaptive_bn import *
import pandas as pd
def rand_prune_and_eval(model, ignore_idx, opt):
    ignore_conv_idx = [i.replace('bn','conv') for i in ignore_idx]
    entropy_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and name not in ignore_conv_idx:
            weights = module.weight.data.cpu().numpy()
            hist, _ = np.histogram(weights.flatten(), bins=50, density=True)
            entropy = np.sum(-hist * np.log2(hist + 1e-7))
            entropy_dict[name] = entropy

    for name in entropy_dict:
        if entropy_dict[name] < -1495.423327:
            entropy_dict[name] = -1495.423327

    min_entropy = np.nanmin(list(entropy_dict.values()))
    max_entropy = np.nanmax(list(entropy_dict.values()))
    normalized_entropy_dict = {name: (entropy - min_entropy) / (max_entropy - min_entropy) for name, entropy in
                               entropy_dict.items()}

    entropy_df = pd.DataFrame(list(entropy_dict.items()), columns=['Layer', 'Entropy'])
    normalized_entropy_df = pd.DataFrame(list(normalized_entropy_dict.items()), columns=['Layer', 'Normalized_Entropy'])

    merged_df = pd.merge(entropy_df, normalized_entropy_df, on='Layer')

    with pd.ExcelWriter('entropy_data.xlsx') as writer:
        merged_df.to_excel(writer, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="runs/train/exp8/weights/best.pt", help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5l-pruning.yaml', help='model.yaml')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--path', type=str, default='models/yolov5l-2024-ours-pruned-0.5.yaml', help='the path to save pruned yaml')
    parser.add_argument('--min_remain_ratio', type=float, default=0.2)
    parser.add_argument('--max_iter', type=int, default=2000, help='maximum number of arch search')
    parser.add_argument('--remain_ratio', type=float, default=0.5, help='the whole parameters/FLOPs remain ratio')
    parser.add_argument('--delta', type=float, default=0.02, help='scale of arch search')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Create model
    model = Model(opt.cfg).to(device)
    ckpt = torch.load(opt.weights, map_location=device)  
    exclude = []                                         # exclude keys
    state_dict = ckpt['model'].float().state_dict()      # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=True)       # load strictly

    # Parse Module
    CBL_idx, ignore_idx, from_to_map = parse_module_defs(model.yaml)
    rand_prune_and_eval(model,ignore_idx,opt)
