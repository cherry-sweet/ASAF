import sys
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from timm.utils import AverageMeter, accuracy, NativeScaler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from tqdm import tqdm
import torch.multiprocessing as mp
from models.build import build_models, freeze_backbone
from setup import config  # 传入设置
from utils.data_loader import build_loader
from utils.eval import *
from utils.info import *
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
import builtins
from settings.setup_functions import *
import torch.distributed as dist
import models.backbone.MIT as MIT

import torch.nn.functional as F

try:
	from torch.utils.tensorboard import SummaryWriter
except:
	pass


def build_model(config, num_classes, rank, log):
	model = build_models(config, num_classes)
	# if torch.__version__[0] == '2' and sys.platform != 'win32':
	# 	# torch.set_float32_matmul_precision('high')
	# 	model = torch.compile(model)
	model.cuda(rank)
	freeze_backbone(model, config.train.freeze_backbone)
	model_without_ddp = model
	n_parameters = count_parameters(model)

	config.defrost()
	config.model.num_classes = num_classes
	config.model.parameters = f'{n_parameters:.3f}M'
	config.freeze()
	if rank in [-1, 0]:
		PSetting(log, 'Model Structure', config.model.keys(), config.model.values(), rank=rank)
		log.save(model)
	return model, model_without_ddp  # (没有分布式的模型)


def main(rank, ngpus_per_node, config):

	if config.distribute.multiprocessing_distributed and rank != 0:  # true
		def print_pass(*args):
			pass
		builtins.print = print_pass

	if rank is not None:
		print("Use GPU: {} for training".format(rank))
	# 打印
	cfg_file = os.path.join('configs', 'swin-cub.yaml')
	print('-' * 18, f'Merge From {cfg_file}'.center(42), '-' * 18)
	print('-' * 18, 'Merge From Argument parser'.center(42), '-' * 18)

	torch.cuda.set_device(rank)
	dist.init_process_group(
		backend=config.distribute.dist_backend,
		init_method=config.distribute.dist_url,
		world_size=config.world_size,
		rank=rank,
	)

	if config.write and rank == 0:
		os.makedirs(config.data.log_path, exist_ok=True)
		with open(config.data.log_path + '/config.json', "w") as f:
			f.write(config.dump())
	log = SetupLogs(config, rank)
	# 初始化进程组
	config.defrost()
	config.local_rank = rank
	config.freeze()

	# Timer
	total_timer = Timer()
	prepare_timer = Timer()
	prepare_timer.start()
	train_timer = Timer()
	eval_timer = Timer()
	total_timer.start()
	# Initialize the Tensorboard Writer
	writer = None
	if config.write:
		try:
			writer = SummaryWriter(config.data.log_path)
		except:
			pass

	# Prepare dataset
	train_loader, test_loader, num_classes, train_samples, test_samples, mixup_fn,_ = build_loader(config)
	step_per_epoch = len(train_loader)
	total_batch_size = config.data.batch_size * get_world_size()
	steps = config.train.epochs * step_per_epoch

	# Build model
	print("=> creating model '{}'".format(config.model.name))
	model, model_without_ddp = build_model(config, num_classes, rank, log)
	# model用于训练  model_without)_ddp:用于保存原有模型
	if config.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],
		                                                  broadcast_buffers=False,
		                                                  find_unused_parameters=False)
	# backbone_low_lr = config.model.type.lower() == 'resnet'
	# optimizer = build_optimizer(config, model, backbone_low_lr)
	optimizer = build_optimizer(config, model, False)
	loss_scaler = NativeScalerWithGradNormCount()
	scheduler = build_scheduler(config, optimizer, step_per_epoch)

	# Determine criterion
	best_acc, best_epoch, train_accuracy = 0., 0., 0.

	if config.data.mixup > 0.:
		criterion = SoftTargetCrossEntropy()
	elif config.model.label_smooth:
		criterion = LabelSmoothingCrossEntropy(smoothing=config.model.label_smooth)
	else:
		criterion = torch.nn.CrossEntropyLoss()

	# Function Mode
	if config.model.resume:
		best_acc = load_checkpoint(config, model, optimizer, scheduler, loss_scaler, log)
		best_epoch = config.train.start_epoch
		accuracy, loss = valid(config, model, test_loader, best_epoch, train_accuracy,writer,True)
		log.info(f'Epoch {best_epoch+1:^3}/{config.train.epochs:^3}: Accuracy {accuracy:2.3f}    '
		         f'BA {best_acc:2.3f}    BE {best_epoch+1:3}    '
		         f'Loss {loss:1.4f}    TA {train_accuracy * 100:2.2f}')
		if config.misc.eval_mode:
			return

	if config.misc.throughput:  # 测量模型的吞吐量
		throughput(test_loader, model, log, config.local_rank)
		return

	# Record result in Markdown Table
	mark_table = PMarkdownTable(log, ['Epoch', 'Accuracy', 'Best Accuracy',
	                                  'Best Epoch', 'Loss'], rank=config.local_rank)

	# End preparation
	torch.cuda.synchronize()
	prepare_time = prepare_timer.stop()
	PSetting(log, 'Training Information',
	         ['Train samples', 'Test samples', 'Total Batch Size', 'Load Time', 'Train Steps',
	          'Warm Epochs'],
	         [train_samples, test_samples, total_batch_size,
	          f'{prepare_time:.0f}s', steps, config.train.warmup_epochs],
	         newline=2, rank=config.local_rank)

	# Train Function
	sub_title(log, 'Start Training', rank=config.local_rank)
	for epoch in range(config.train.start_epoch, config.train.epochs):
		train_timer.start()
		if config.local_rank != -1:
			train_loader.sampler.set_epoch(epoch)
		# list1 = list(model.named_parameters())
		# print(list1[76])

		if not config.misc.eval_mode:
			train_accuracy = train_one_epoch(config, model, criterion, train_loader, optimizer,
			                                 epoch, scheduler, loss_scaler, mixup_fn, writer)
		train_timer.stop()

		# Eval Function
		eval_timer.start()
		if (epoch + 1) % config.misc.eval_every == 0 or epoch + 1 == config.train.epochs:
			accuracy, loss = valid(config, model, test_loader, epoch, train_accuracy, writer,False)
			if config.local_rank in [-1, 0]:
				if best_acc < accuracy:
					best_acc = accuracy
					best_epoch = epoch + 1
					if config.write and epoch > 1 and config.train.checkpoint:
						save_checkpoint(config, epoch, model, best_acc, optimizer, scheduler, loss_scaler, log)
				log.info(f'Epoch {epoch + 1:^3}/{config.train.epochs:^3}: Accuracy {accuracy:2.3f}    '
				         f'BA {best_acc:2.3f}    BE {best_epoch:3}    '
				         f'Loss {loss:1.4f}    TA {train_accuracy * 100:2.2f}')
				if config.write:
					mark_table.add(log, [epoch + 1, f'{accuracy:2.3f}',
					                     f'{best_acc:2.3f}', best_epoch, f'{loss:1.5f}'], rank=config.local_rank)
			pass  # Eval
		eval_timer.stop()
		pass  # Train

	# Finish Training
	if writer is not None:
		writer.close()
	train_time = train_timer.sum / 60
	eval_time = eval_timer.sum / 60
	total_time = train_time + eval_time
	total_time_true = total_timer.stop()
	total_time_true = total_time_true/60
	PSetting(log, "Finish Training",
	         ['Best Accuracy', 'Best Epoch', 'Training Time', 'Testing Time', 'Syncthing Time','Total Time'],
	         [f'{best_acc:2.3f}', best_epoch, f'{train_time:.2f} min', f'{eval_time:.2f} min', f'{total_time_true-total_time:.2f} min' ,f'{total_time_true:.2f} min'],
	         newline=2, rank=config.local_rank)


def train_one_epoch(config, model, criterion, train_loader, optimizer, epoch, scheduler, loss_scaler, mixup_fn=None,
                    writer=None, MIT = True):
	model.train()
	optimizer.zero_grad()

	step_per_epoch = len(train_loader)
	loss_meter = AverageMeter()
	norm_meter = AverageMeter()
	scaler_meter = AverageMeter()
	epochs = config.train.epochs
	# 这三个loss分别代表什么作用
	loss1_meter = AverageMeter()
	loss2_meter = AverageMeter()
	loss3_meter = AverageMeter()

	p_bar = tqdm(total=step_per_epoch,
	             desc=f'Train {epoch + 1:^3}/{epochs:^3}',
	             dynamic_ncols=True,
	             ascii=True,
	             disable=config.local_rank not in [-1, 0])
	all_preds, all_label = None, None
	for step, (x, y,x1) in enumerate(train_loader):
		global_step = epoch * step_per_epoch + step
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
		x1 = x1.cuda(non_blocking=True)
		label = y.clone()

		if mixup_fn:
			x, y = mixup_fn(x, y)
			x1,label = mixup_fn(x1,label)

		with torch.cuda.amp.autocast(enabled=config.misc.amp):
			if config.model.baseline_model:  # baseline
				logits = model(x,epoch)   # (16,200)
				logits1 = model(x1, epoch)

			else:
				logits = model(x, y)
		# logits list4:(16,200)
		logits, loss, other_loss = loss_in_iters(logits, y, criterion,logits1)  # 对于baseline来说，第一项没变，第二项是平滑交叉熵损失
		# 训练精度 用的是融合的特征表示
		is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
		grad_norm = loss_scaler(loss, optimizer, clip_grad=config.train.clip_grad,
		                        parameters=model.parameters(), create_graph=is_second_order)

		optimizer.zero_grad()
		scheduler.step_update(global_step + 1)
		loss_scale_value = loss_scaler.state_dict()["scale"]
		if mixup_fn is None:  # 计算准确度  改成多个
			# preds = torch.argmax(logits, dim=-1)  # (16)
			# all_preds, all_label = save_preds(preds, y, all_preds, all_label)
			preds = []
			for i in range(5):
				pred = torch.argmax(logits[i],dim=-1)
				preds.append(pred)
			all_preds, all_label = save_preds_five(preds, y, all_preds, all_label)
		torch.cuda.synchronize()

		if grad_norm is not None:
			norm_meter.update(grad_norm)
		scaler_meter.update(loss_scale_value)
		loss_meter.update(loss.item(), y.size(0))

		lr = optimizer.param_groups[0]['lr']
		if writer:
			writer.add_scalar("train/loss", loss_meter.val, global_step)
			writer.add_scalar("train/lr", lr, global_step)
			writer.add_scalar("train/grad_norm", norm_meter.val, global_step)
			writer.add_scalar("train/scaler_meter", scaler_meter.val, global_step)
			if other_loss:
				try:
					loss1_meter.update(other_loss[0].item(), y.size(0))
					loss2_meter.update(other_loss[1].item(), y.size(0))
					loss3_meter.update(other_loss[2].item(), y.size(0))
				except:
					pass

				writer.add_scalar("losses/t_loss", loss_meter.val, global_step)
				writer.add_scalar("losses/1_loss", loss1_meter.val, global_step)
				writer.add_scalar("losses/2_loss", loss2_meter.val, global_step)
				writer.add_scalar("losses/3_loss", loss3_meter.val, global_step)

		# set_postfix require dic input
		p_bar.set_postfix(loss="%2.5f" % loss_meter.avg, lr="%.5f" % lr, gn="%1.4f" % norm_meter.avg)
		p_bar.update()

	# After Training an Epoch
	p_bar.close()  # all_preds:(1488)  all_label:(1488)  这应该是不同卡上的合并
	train_accuracy = eval_accuracy_five(all_preds, all_label, config) if mixup_fn is None else 0.0
	if mixup_fn is None:
		print( '[epoch {}] ACC: {:.4f} {:.4f} {:.4f} {:.4f} || {:.4f}'.format(epoch, train_accuracy[0], train_accuracy[1], train_accuracy[2], train_accuracy[3],
																		  train_accuracy[4]))
		train_accuracy=train_accuracy[-1]
	return train_accuracy


def con_loss(features, labels):
	B, _ = features.shape
	features = F.normalize(features)
	cos_matrix = features.mm(features.t())
	pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
	neg_label_matrix = 1 - pos_label_matrix
	pos_cos_matrix = 1 - cos_matrix
	neg_cos_matrix = cos_matrix - 0.4
	neg_cos_matrix[neg_cos_matrix < 0] = 0
	loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
	loss /= (B * B)
	return loss



def loss_in_iters(output, targets, criterion,logit1=None,label = None):
	if isinstance(output, list):
		if len(output)==4:
			loss = 0.2*criterion(output[0], targets)+0.3*criterion(output[1], targets)+0.4*criterion(output[2], targets)+ criterion(output[3], targets) + 0*criterion(logit1[3], targets)
			out = 0.2*output[0]+0.3*output[1]+0.4*output[2]+output[3]
			output.append(out)
		else:
			loss = criterion(output, targets)
		return output,loss,None
	if not isinstance(output, (list, tuple)):
		return output, criterion(output, targets), None
	else:
		logits, loss = output
		if not isinstance(loss, (list, tuple)):
			return logits, loss, None
		else:
			return logits, loss[0], loss[1:]

@torch.no_grad()
def valid(config, model, test_loader, epoch=-1, train_acc=0.0, writer=None,save_feature=False):
	criterion = torch.nn.CrossEntropyLoss()
	model.eval()

	step_per_epoch = len(test_loader)
	p_bar = tqdm(total=step_per_epoch,
	             desc=f'Valid {(epoch + 1) // config.misc.eval_every:^3}/{math.ceil(config.train.epochs / config.misc.eval_every):^3}',
	             dynamic_ncols=True,
	             ascii=True,
	             disable=config.local_rank not in [-1, 0])

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	# acc_meters = [AverageMeter() for _ in range(5)]
	saved_feature,saved_labels = [],[]
	for step, (x, y) in enumerate(test_loader):
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

		with torch.cuda.amp.autocast(enabled=config.misc.amp):
			logits = model(x)
			if isinstance(logits, list):
				if len(logits) == 4:
					out =  logits[3] #0.002 * logits[0] + 0.003 * logits[1] + 0.004 * logits[2] + logits[3]
					logits = out
				else:
					logits = logits
		if save_feature:
			saved_feature.append(logits)
			saved_labels.append(y)

		loss = criterion(logits, y.long())
		acc = accuracy(logits, y)[0]
		# acc = accuracy_multi(logits, y)  #  改了这里  应该返回的是列表中的列表
		if config.local_rank != -1:
			acc = reduce_mean(acc)  # 改了这里

		loss_meter.update(loss.item(), y.size(0))
		acc_meter.update(acc.item(), y.size(0))
		# for i, acc in enumerate(acc):
		# 	acc_meters[i].update(acc.item(), y.size(0))

		p_bar.set_postfix(acc="{:2.3f}".format(acc_meter.avg), loss="%2.5f" % loss_meter.avg,
		                  tra="{:2.3f}".format(train_acc * 100))
		p_bar.update()
		pass

	if save_feature:
		os.makedirs('visualize/saved_features',exist_ok=True)
		saved_feature = torch.cat(saved_feature, 0)
		saved_labels = torch.cat(saved_labels,0)
		torch.save(saved_feature,f'visualize/saved_features/{config.data.dataset}_f.pth')
		torch.save(saved_labels, f'visualize/saved_features/{config.data.dataset}_l.pth')
	p_bar.close()
	if writer:
		writer.add_scalar("test/accuracy", acc_meter.avg, epoch + 1)
		writer.add_scalar("test/loss", loss_meter.avg, epoch + 1)
		writer.add_scalar("test/train_acc", train_acc * 100, epoch + 1)
	return acc_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, log, rank):
	model.eval()
	for idx, (images, _) in enumerate(data_loader):
		images = images.cuda(non_blocking=True)
		batch_size = images.shape[0]
		for i in range(50):
			model(images)
		torch.cuda.synchronize()
		if rank in [-1, 0]:
			log.info(f"throughput averaged with 30 times")
		tic1 = time.time()
		for i in range(30):
			model(images)
		torch.cuda.synchronize()
		tic2 = time.time()
		if rank in [-1, 0]:
			log.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
		return


if __name__ == '__main__':
	ngpus_per_node = torch.cuda.device_count()  # 找进程
	if config.distribute.multiprocessing_distributed:
		mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
	else:
		main(None, ngpus_per_node, config)
