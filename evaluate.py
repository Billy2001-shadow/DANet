import os
import torch
from tqdm import tqdm
from options import get_args
from dataloader import dataloader
from models.net import DANet
from utils import *

os.environ['CUDA_VISIBLE_DEVICES']='0'
args = get_args('test')
# lode nyudv2 test set
TestImgLoader = dataloader.getTestingData_NYUDV2(1, args.testlist_path, args.root_path)
# model

model = DANet(args)
model = torch.nn.DataParallel(model)
model.cuda()
model.eval()

metrics =RunningAverageDict()

def test(model):
    print("loading model {}".format(args.loadckpt))
    state_dict=torch.load(args.loadckpt)["model"]
    model.load_state_dict(state_dict)
   
    with torch.no_grad():
        metrics =RunningAverageDict()

        for batch_idx, sample in tqdm(enumerate(TestImgLoader),total=len(TestImgLoader)):
            img = sample['image'].cuda()
            depth = sample['depth'].cuda()

            bin_edges,output= model(img)
            pred=torch.nn.functional.interpolate(output[-1], size=[depth.size(2), depth.size(3)],mode='bilinear', align_corners=True)
            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth] = args.min_depth
            pred[pred > args.max_depth] = args.max_depth
            pred[np.isinf(pred)] = args.max_depth
            pred[np.isnan(pred)] = args.min_depth

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask=np.logical_and(gt_depth>args.min_depth, gt_depth<args.max_depth)
            metrics.update(compute_errors(gt_depth[valid_mask], pred[valid_mask]))
      
        metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
        print(f"Metrics: {metrics}")
# def test(model):
#     print("loading model {}".format(args.loadckpt))
#     state_dict=torch.load(args.loadckpt)["model"]
#     model.load_state_dict(state_dict)
#     totalNumber=0

#     errorSum={'MSE': 0, 'RMSE': 0,'RMSE_FD': 0, 'ABS_REL': 0, 'LG10': 0,
#               'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
#     for batch_idx, sample in tqdm(enumerate(TestImgLoader),total=len(TestImgLoader)):
#         image, depth  = sample['image'], sample['depth']
#         depth = depth.cuda()
#         image = image.cuda()
       
#         with torch.no_grad():
#             bin,output= model(image)
#             output=torch.nn.functional.interpolate(output[-1], size=[depth.size(2), depth.size(3)],mode='bilinear', align_corners=True)
#             valid_mask=torch.logical_and(depth>args.min_depth, depth<args.max_depth)
#             totalNumber=totalNumber+depth.size(0) # 一共处理了多少图片
#             errors=evaluateError(output[valid_mask], depth[valid_mask])
#             errorSum=addErrors(errorSum, errors, depth.size(0))
#             averageError=averageErrors(errorSum, totalNumber)
#     averageError['RMSE']=np.sqrt(averageError['MSE'])

#     for k,v in averageError.items():
#         print(k,v)

if __name__ == '__main__':
    test(model)
