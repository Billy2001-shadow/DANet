import random
import wandb
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.parallel
from utils import *
import model_io
from options import get_args
from models.net import DANet
from dataloader import dataloader
from models.loss import *
PROJECT = "MDE-DANet"
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
args = get_args('train')
args.name=f"{args.backbone}+bs={args.batch_size}"
args.checkpoint_dir=os.path.join("./results",args.name)  # 把训练的参数全写入到一个文件中保存 lr batch_size epochs等
makedir(args.checkpoint_dir)

# 确保训练过程的随机性可控，实验结果具有可复现性。
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)  # Numpy module.
random.seed(args.seed)  # Python random module.
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True

#losses
criterion_depth = Total_loss()
criterion_ueff=SILogLoss()        # 对数尺度损失，用于优化深度估计的相对误差。
criterion_bins=BinsChamferLoss() if args.w_chamfer>0 else None #对分箱中心与真实深度分布对齐的约束损失（如果启用 w_chamfer）。
criterion_minmax=MinmaxLoss() if args.w_minmax>0 else None     #对分箱中心的最小值和最大值的正则化损失。

# dataset, dataloader
TrainImgLoader = dataloader.getTrainingData_NYUDV2(args.batch_size, args.trainlist_path, args.root_path)
TestImgLoader = dataloader.getTestingData_NYUDV2(1, args.testlist_path, args.root_path)
# model, optimizer
model = DANet(args)
model = nn.DataParallel(model)
model.cuda()

optimizer = build_optimizer(model = model,
			    learning_rate=args.lr,
			    optimizer_name=args.optimizer_name,
			    weight_decay = args.weight_decay,
			    epsilon=args.epsilon,
			    momentum=args.momentum
			    )

# load parameters
start_epoch = 0
## progress
if args.resume:
	loadckpt =args.loadckpt
	print("loading the lastest model in loadckpt: {}".format(loadckpt))
	state_dict = torch.load(loadckpt)
	model.load_state_dict(state_dict)
	start_epoch=state_dict["epoch"]
elif args.loadckpt is not None:
	print("loading model {}".format(args.loadckpt))
	state_dict = torch.load(args.loadckpt)["model"]
	model.load_state_dict(state_dict)
else:
	print("start at epoch {}".format(start_epoch))

## train process
def train():
	global PROJECT,criterion_ueff
	experiment_name=args.name
	print(f"Training {experiment_name}")
	run_id=f"nodebs{args.batch_size}-tep{args.epochs}-lr{args.lr}"
	name=f"{experiment_name}_{run_id}"

	if args.logging:
		tags=args.tags.split(',') if args.tags!='' else None
		wandb.init(project=PROJECT, name=name, config=args, dir=args.checkpoint_dir, tags=tags, notes=args.notes,mode="offline")

	# some globals
	iters=len(TrainImgLoader) # 一共有iters个batch
	step=start_epoch*iters
	best_loss=0
	for epoch in range(start_epoch, args.epochs):
		if epoch==10:
			args.lr*=100
			criterion_ueff=SILogLoss_valley(factor=args.valley_factor)
		if args.logging: wandb.log({"Epoch": epoch}, step=step)
		adjust_learning_rate(optimizer, epoch, args.lr)
		model.train()
		# 每个batch进行一次参数更新
		for i, sample in tqdm(enumerate(TrainImgLoader), desc=f"Epoch: {epoch+1}/{args.epochs}. Loop: Train",total=len(TrainImgLoader)):
			image, depth = sample['image'], sample['depth']

			depth = depth.cuda()
			image = image.cuda()
			optimizer.zero_grad() # 梯度清零

			bin_edges,pred_depth = model(image)

			gt_depth = adjust_gt(depth, pred_depth)
			#Calculate the total loss
			l_dense=criterion_ueff(pred_depth, gt_depth)
			if args.w_chamfer>0:
				l_chamfer=criterion_bins(bin_edges, depth)
			else:
				l_chamfer=torch.Tensor([0]).to(image.device)

			if args.w_minmax>0 and epoch>=10:
				l_minmax=criterion_minmax(bin_edges,depth)
			else:
				l_minmax=torch.Tensor([0]).to(image.device)

			loss=l_dense+args.w_chamfer*l_chamfer+args.w_minmax*l_minmax
			loss.backward()	 # 反向传播
			optimizer.step() # 更新参数

			if args.logging and step%5==0:
				wandb.log({f"Train/loss": loss.item()}, step=step)
				wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
				if args.w_chamfer>0:
					wandb.log({f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)
				if args.w_minmax>0:
					wandb.log({f"Train/{criterion_minmax.name}": l_minmax.item()}, step=step)
			step+=1

			if step%args.validate_every==0:
				# print(f"Step {step}, Loss: {loss.item()}") # 输出当前训练集的损失
				################################# Validation loop ##################################################
				model.eval() # 模型暂时切换到评估模式
				metrics=validate(args, model, TestImgLoader, epoch, args.epochs)

				print("Validated: {}".format(metrics))
				if args.logging:
					wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
					model_io.save_checkpoint(model, optimizer, epoch,step, f"{experiment_name}_{run_id}_latest.pt",
											 root=os.path.join(args.checkpoint_dir, "checkpoints"))

				if metrics['a1']>best_loss:
					model_io.save_checkpoint(model, optimizer, epoch,step, f"{experiment_name}_{run_id}_best.pt",
											 root=os.path.join(args.checkpoint_dir, "checkpoints"))
					best_loss=metrics['a1']
				model.train() # 评估完成后再切换回训练模式 
			#################################################################################################
		# if epoch>=5:
		# 	model_io.save_checkpoint(model, optimizer, epoch, step, f"{epoch}.pt",
		# 							 root=os.path.join(args.checkpoint_dir, "checkpoints"))
def validate(args, model, test_loader, epoch, epochs):
	with torch.no_grad():
		metrics =RunningAverageDict()
		for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation"):
			img = batch['image'].cuda()
			depth = batch['depth'].cuda()

			depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
			bin_edges,output = model(img)
			pred = nn.functional.interpolate(output[-1], depth.shape[-2:], mode='nearest') #  bilinear , align_corners=True
			pred = pred.squeeze().cpu().numpy()
			# 对预测出来的值进行后处理
			pred[pred < args.min_depth] = args.min_depth
			pred[pred > args.max_depth] = args.max_depth
			pred[np.isinf(pred)] = args.max_depth
			pred[np.isnan(pred)] = args.min_depth

			gt_depth = depth.squeeze().cpu().numpy()
			valid_mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)
			metrics.update(compute_errors(gt_depth[valid_mask], pred[valid_mask]))

		return metrics.get_value()

if __name__ == '__main__':
	train()


