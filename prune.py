import csv
from models.yolo import *
from utils.general import set_logging
from utils.torch_utils import select_device
from utils.prune_utils import *
from utils.adaptive_bn import *

def rand_prune_and_eval(model, ignore_idx, opt):
    origin_flops = model.flops
    ignore_conv_idx = [i.replace('bn','conv') for i in ignore_idx]
    candidates = 0
    max_mAP = 0
    maskbndict = {}
    maskconvdict = {}
    with open(opt.cfg) as f:
        oriyaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

    ABE = AdaptiveBNEval(model, opt, device, hyp)
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
    print(min_entropy)
    print(max_entropy)
    print("entropy values: ")
    for it in list(entropy_dict.values()):
        print(it)
    normalized_entropy_dict = {name: (entropy - min_entropy) / (max_entropy - min_entropy) for name, entropy in
                               entropy_dict.items()}

    while True:
        pruned_yaml = deepcopy(oriyaml)

        # obtain mask
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if name in ignore_conv_idx:
                    mask = torch.ones(module.weight.data.size()[0]).to(device) # [N, C, H, W]
                else:

                    upper_limit = min(1, normalized_entropy_dict[name] + 0.2)
                    lower_limit = max(0.2,normalized_entropy_dict[name] - 0.4)
                    rand_remain_ratio = max(0.2, random.uniform(lower_limit,upper_limit) - 0.1)
                    mask = obtain_filtermask_l1(module, rand_remain_ratio).to(device)
                maskbndict[(name[:-4] + 'bn')] = mask
                maskconvdict[name] = mask

        pruned_yaml = update_yaml(pruned_yaml, model, ignore_conv_idx, maskconvdict, opt)

        compact_model = Model(pruned_yaml, pruning=True).to(device)
        current_flops = compact_model.flops
        print(current_flops/origin_flops)
        if (current_flops/origin_flops > opt.remain_ratio+opt.delta) or (current_flops/origin_flops < opt.remain_ratio-opt.delta):
            del compact_model
            del pruned_yaml
            continue
        weights_inheritance(model, compact_model, from_to_map, maskbndict)
        mAP = ABE(compact_model)
        tmp = []
        tmp.append(mAP)
        with open('ours_2024/candidates_' + str(opt.remain_ratio) + '_ours.csv', "a", newline='') as file:
            f_csv = csv.writer(file)
            f_csv.writerow(tmp)
        print('mAP@0.5 of candidate sub-network is {:f}'.format(mAP))

        if mAP > max_mAP:
            max_mAP = mAP
            with open(opt.path, "w", encoding='utf-8') as f:
                yaml.safe_dump(pruned_yaml,f,encoding='utf-8', allow_unicode=True, default_flow_style=True, sort_keys=False)
            ckpt = {'epoch': -1,
                    'best_fitness': [max_mAP],
                    'model': deepcopy(de_parallel(compact_model)).half(),
                    'ema': None,
                    'updates': None,
                    'optimizer': None,
                    'wandb_id': None}
            torch.save(ckpt, 'ours_2024/pruned_' + str(opt.remain_ratio) + '_ours.pt')

        candidates = candidates + 1
        del compact_model
        del pruned_yaml
        if candidates > opt.max_iter:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="runs/train/exp/weights/best.pt", help='initial weights path')
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
