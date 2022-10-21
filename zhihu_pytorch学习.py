# 1. 自动混合精度 [torch.__version__ >=1.6]
# [1]https://zhuanlan.zhihu.com/p/165152789
# [2]https://zhuanlan.zhihu.com/p/79887894
# [3]https://link.zhihu.com/?target=https%3A//docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
# [4]https://link.zhihu.com/?target=https%3A//nvidia.github.io/apex/amp.html
# [5]https://link.zhihu.com/?target=http%3A//on-demand.gputechconf.com/gtc-taiwan/2018/pdf/5-1_Internal%2520Speaker_Michael%2520Carilli_PDF%2520For%2520Sharing.pdf
##1. 什么是自动混合精度

##2. 为什么需要自动混合精度?   

##3. 怎么使用pytorch的自动混合精度



###1. 
'''
torch中默认数据类型: 32位浮点型, 为了节省显存也好, 加速计算也好. 混合精度就是允许模型中不只float32一种数据精度.

'''

###2.
from torch.cuda.amp import autocast as autocast
'''
既然是torch.cuda前缀且是NVIDIA开发的, 故只有支持Tensor core(Tensor Core每个时钟执行64个浮点混合精度操作, Tensor Core进行矩阵运算可轻易提速, 降低一半的显存访问和存储)的CUDA硬件才能amp.
[具体那些gpu支持呢? 可NVIDIA官网查看. 举点栗子: 支持: 2080Ti, Titan, Tesla等, 不支持: Pascal系列]
amp中允许的精度: torch.FloatTensor 和 torch.HalfTensor(即float32, 16)
自动代表, Tensor的dtype会自动变化(自动调整). [需辅助一点, 手工干预]
Q: 为什么要torch.HalfTensor半精度?
A: 节省显存, 比如batchsize就可开的更大; 训练计算得更快
torch.HalfTensor的缺点: 数值范围小, 则可能出现溢出.. 精度丢失带来误差Rounding Error(2-3 + 2-13 -> 2-3 丢失精度了). 
针对torch.HalfTensor缺点的补救方法: 
1. torch.cuda.amp.GradScaler做梯度scale. 放大loss防止梯度underflow, 但记得更新参数权重的时候, 把梯度unscale回去.
2. 部分"场景"下恢复FloatTensor精度, 由pytorch框架决定.

会自动torch.HalfTensor的场景:
__matmul__
addbmm
addmm
addmv
addr
baddbmm
bmm
chain_matmul
conv1d
conv2d
conv3d
conv_transpose1d
conv_transpose2d
conv_transpose3d
linear
matmul
mm
mv
prelu


###3. 
上文已经提及了, 实现自动混合精度: autocast + GradScaler, 代码示例:

from torch.cuda.amp import autocast as autocast

# model默认torch.FloatTensor精度
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# 在训练最开始之前实例化一个GradScaler对象
scaler = GradScaler()

for input, target in data:
    optimizer.zero_grad()
    # 前向过程(model + loss)开启autocast
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)
	
	# autocast之外了~  scale loss放大梯度
    scaler.scale(loss).backward()

    # scaler.step(optimizer)把梯度值unscale回来, 并检查梯度是否inf或NaN
    scaler.step(optimizer)
    
    # 准备着看是否要增大scaler
    scaler.update()  # 当连续多次(growth_interval指定)没出现梯度inf或NaN，则scaler.update()会将scaler增大: scaler *= growth_factor
	
	# 如果梯度的值不是infs或NaNs, 调用optimizer.step()更新权重; 
	# 否则会忽略optimizer.step()调用, 保证权重不更新(不被破坏). 还会做scaler *= backoff_factor, 来缩小scaler值
    loss.backward()
    optimizer.step()

在autocast上下文内, pytorch中那些会自动变为HalfTensor的ops就转为半精度运算了.  


最后补充下, 自动混合精度等级:
from apex import amp
model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 欧一
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()

# O0: 纯FP32训练, 可以作为accuracy的baseline;
# O1: 混合精度训练, pytorch支持的自动混合精度op会自动半精度处理;
# O2: "几乎FP16"混合精度训练, 除了Batch norm几乎都用FP16;
# O3: 纯FP16训练, 很不稳定但可作为speed的baseline



# 2. 优化器
优化器optimzier的作用: 根据网络反向传播的梯度信息来更新网络参数, 以起到降低loss值的作用. 故我们需要在优化器中传入model参数..
step函数使用参数空间(param_groups)中的grad, 这即解释了为什么optimzier使用前需要zero清零(避免min-batch之间梯度累加). 另外优化器作用在反向传播时候的loss上, 故需先一步: loss.backward().

optimizer.zero_grad()   #optimizer梯度清0
loss = loss_fn(outputs, y_train)  #计算loss
loss.backward()       #loss反向传播
optimizer.step()       #根据loss反向梯度optimizer进行参数更新

例举一些优化器的, step(): SGD, Adam等.
[6]https://zhuanlan.zhihu.com/p/87209990



# 3. dataloader  多gpu分布式dataloader 
[7]https://zhuanlan.zhihu.com/p/30934236
[8]https://zhuanlan.zhihu.com/p/80695364
pytorch数据读取主要涉及以下三个类: 
1. Dataset: 包含__getitem__, __len__
def __getitem__(self, index):
    img_path, label = self.data[index].img_path, self.data[index].label
    img = Image.open(img_path)

    return img, label

def __len__(self):
    return len(self.data)


2. DataLoader:
torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)

dataset: 即上文介绍的Dataset
collate_fn: 用来打包batch的函数
num_worker: 多线程方法, 设置为>=1即可多线程读取数据 

完成的事情:
1.定义许多成员变量, 后续可赋值给DataLoaderIter; 2.实现__iter__()函数, 把自己"装进"DataLoaderIter.

3. DataLoaderIter
torch.utils.data.dataloader.DataLoaderIter. 支持定义自己的数据集, 制作为dataset.
class CustomDataset(Dataset):
   # 自定义自己的dataset

dataset = CustomDataset()
dataloader = torch.utils.data.DataLoader(dataset, ...)

for data in dataloader:
	1. 调用DataLoader的__iter__()方法, 产生了一个DataLoaderIter
	2. 反复调用DataLoaderIter的__next__()来得到batch: 多次调用dataset的__getitem__()方法(num_worker>=1就是多线程调用), 用collate_fn来把它们打包成batch. 还可设置shuffle, sample等方法.
	3. 数据读完后, __next__()抛出一个StopIteration异常, for循环结束dataloader失效

另外还有加速数据读取的sao操作:
1. prefetch_generator在后台加载下一batch的数据. [pip install prefetch_generator]
实现DataLoaderX类, 替换原本的DataLoader类读取数据.

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

提速原因: PyTorch默认的DataLoader会创建一些worker线程来预读取新的数据, 但除非这些线程的数据全部都被清空, 这些线程才会读下一batch数据. 故使用prefetch_generator保证线程不等待, 保证每个线程都总有至少一个数据在加载.

2. 使用data_prefetcher新开cuda stream来拷贝tensor到gpu.
class DataPrefetcher():
    def __init__(self, loader, opt):
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

# traing code:
# 实用实现的DataPrefetcher类读取数据
prefetcher = DataPrefetcher(data_loader, opt)
batch = prefetcher.next()
iter_id = 0
while batch is not None:
    iter_id += 1
    if iter_id >= num_iters:
        break
    run_step()
    batch = prefetcher.next()

提速原因: PyTorch将所有涉及到GPU的操作(比如内核操作cpu->gpu, gpu->cpu)都排入同一个stream中, 无法并行. 当当前前向在default stream中, 则下一个batch数据(需要cpu->gpu的话)必须在另一个stream中.
注意dataloader设置pin_memory=True即可.

3. num_worker设置合理, 不宜太大或太小. 太小不能充分利用多线程提速, 太大会造成线程阻塞或撑爆内存.

others: 使用apex.DistributedDataParallel替代torch.DataParallel; 使用apex加速



# 4. 多GPU训练






# 5. 训练tips







龚总好写完 都同步一份到知乎  引流  