# pytorch常见代码段, 训练tips
#[1]https://zhuanlan.zhihu.com/p/59205847
#[2]https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

# Benchmark模式可提升计算速度, 但由于计算有随机性每次网络前馈结果略有差异
torch.backends.cudnn.benchmark = True
# 想避免这种结果波动, 设置:
torch.backends.cudnn.deterministic = True

# 清除GPU存储: Control+C终止运行后GPU存储未及时释放, 在PyTorch内可:
torch.cuda.empty_cache() 

# 从只包含一个元素的张量中提取值. 统计训练loss时常用, [否则将累积计算图, GPU存储占用越来越大]
loss_value = loss_tensor.item()

# tensor打乱顺序
tensor2 = tensor1[torch.randperm(tensor1.size(0))]  # Shuffle the first dimension

# 复制操作
# Operation                 |  New/Shared memory | Still in computation graph |
tensor.clone()            # |        New         |          Yes               |
tensor.detach()           # |      Shared        |          No                |
tensor.detach.clone()()   # |        New         |          No                |
# detach()函数基本就是把原先的节点从graph中剔除了的..~

# torch.cat沿着给定维度拼接, torch.stack会新增一维
>>> a
tensor([[-0.1888],
        [-0.8155]])
>>> b
tensor([[0.5386],
        [2.2672]])
>>> c = torch.cat([a,b], dim=0)
>>> c
tensor([[-0.1888],
        [-0.8155],
        [ 0.5386],
        [ 2.2672]])
>>> d = torch.stack([a,b], dim=0)
>>> d
tensor([[[-0.1888],
         [-0.8155]],

        [[ 0.5386],
         [ 2.2672]]])
>>> d.size()
torch.Size([2, 2, 1])
>>> c.size()
torch.Size([4, 1])

# 整数转为one_hot编码形式, num_class=5, N=3
>>> tensor= torch.randint(0,5, [3])
>>> tensor
tensor([2, 0, 4])
>>> tensor.size()
torch.Size([3])
>>> one_hot = torch.zeros(3, 5).long()
>>> one_hot
tensor([[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]])
# .scatter_ 注意这个赋值函数
>>> one_hot.scatter_(dim=1,index=tensor.unsqueeze(dim=1),src=torch.ones(3, 5).long())
tensor([[0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]])

# 矩阵乘法
# Matrix multiplication: (m*n) * (n*p) -> (m*p).
result = torch.mm(tensor1, tensor2)

# Batch matrix multiplication: (b*m*n) * (b*n*p) -> (b*m*p).
result = torch.bmm(tensor1, tensor2)

# Element-wise multiplication.
result = tensor1 * tensor2  # 也即, torch.mul(a,b)

# 多卡间同步BN
sync_bn = torch.nn.SyncBatchNorm(num_features, eps=1e-05, momentum=0.1, affine=True, 
                               track_running_stats=True)
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

# 参数初始化 params initial
for layer in model.modules():
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                      nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

# Initialization with given tensor.
layer.weight = torch.nn.Parameter(tensor)
# model.modules()迭代遍历模型的所有子层, model.children()只会遍历模型下的一层

# 使用预训练模型
model.load_state_dict(torch.load('model.pth'), strict=False)

# gpu模型转到cpu上使用
model.load_state_dict(torch.load('model.pth', map_location='cpu'))

# 提取模型某一层的feature map
# VGG-16 relu5-3 feature.
model = torchvision.models.vgg16(pretrained=True).features[:-1]
# VGG-16 pool5 feature.
model = torchvision.models.vgg16(pretrained=True).features
# VGG-16 fc7 feature.
model = torchvision.models.vgg16(pretrained=True)
model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])
# ResNet GAP feature.
model = torchvision.models.resnet18(pretrained=True)
model = torch.nn.Sequential(collections.OrderedDict(
    list(model.named_children())[:-1]))

# model.eval()然后输入image得到输出, 即为feature map
with torch.no_grad():
    model.eval()
    conv_representation = model(image)

# 提取多层feature map: .no_grad() + eval()
class FeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained_model, layers_to_extract):
        torch.nn.Module.__init__(self)
        self._model = pretrained_model
        self._model.eval()
        self._layers_to_extract = set(layers_to_extract)
    
    def forward(self, x):
        with torch.no_grad():
            conv_representation = []
            for name, layer in self._model.named_children():
                x = layer(x)
                if name in self._layers_to_extract:
                    conv_representation.append(x)
            return conv_representation


# 区分层设置学习率: fc 和 conv 层分别设置学习率
model = torchvision.models.resnet18(pretrained=True)
finetuned_parameters = list(map(id, model.fc.parameters()))
conv_parameters = (p for p in model.parameters() if id(p) not in finetuned_parameters)
parameters = [{'params': conv_parameters, 'lr': 1e-3}, 
              {'params': model.fc.parameters()}]
optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)

# 不对偏置项进行L2正则化/权值衰减(weight decay)
bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
parameters = [{'parameters': bias_list, 'weight_decay': 0},                
              {'parameters': others_list}]
optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)
# 上面俩都是在parameters{}中做设置


# label smoothing
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    N = labels.size(0)
    # C is the number of classes.
    smoothed_labels = torch.full(size=(N, C), fill_value=0.1 / (C - 1)).cuda()
    # 又见scatter_()赋值函数
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=0.9)

    score = model(images)
    log_prob = torch.nn.functional.log_softmax(score, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# mixup: 包括图像和loss
beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()

    # Mixup images.
    lambda_ = beta_distribution.sample([]).item()
    index = torch.randperm(images.size(0)).cuda()
    mixed_images = lambda_ * images + (1 - lambda_) * images[index, :]

    # Mixup loss.    
    scores = model(mixed_images)
    loss = (lambda_ * loss_function(scores, labels) 
            + (1 - lambda_) * loss_function(scores, labels[index]))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)

# 模型保存于断点tune
# Save checkpoint
is_best = current_acc > best_acc
best_acc = max(best_acc, current_acc)
checkpoint = {
    'best_acc': best_acc,    
    'epoch': t + 1,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
model_path = os.path.join('model', 'checkpoint.pth.tar')
torch.save(checkpoint, model_path)
if is_best:
    shutil.copy('checkpoint.pth.tar', model_path)

# Load checkpoint
if resume:
    model_path = os.path.join('model', 'checkpoint.pth.tar')
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Load checkpoint at epoch %d.' % start_epoch)

# 统计代码各部分耗时
with torch.autograd.profiler.profile(enabled=True, use_cuda=False) as profile:
    ...
print(profile)  # or python -m torch.utils.bottleneck main.py


# pytorch训练tips
# 运行前向时开启异常检测功能, 则反向时会打印引起反向失败的前向操作堆栈; 反向计算时出现nan也会异常
# 缺陷: 会降低训练速度, 谨慎使用.
torch.autograd.detect_anomaly()




