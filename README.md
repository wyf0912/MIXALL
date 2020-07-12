# Heterogeneous Domain Generalization via Domain Mixup
The code release of paper 'Heterogeneous Domain Generalization via Domain Mixup' ICASSP 2020.

A **simple** but **effective** way to improve the heterogeneous domain generalization performance. The core code is as follows.

The [paper](https://github.com/wyf0912/Heterogeneous-Domain-Generalization-via-Domain-Mixup/blob/master/HETEROGENEOUS%20DOMAIN%20GENERALIZATION%20VIA%20DOMAIN%20MIXUP.pdf) can be found here.

```python
def mixall(x, y, beta=8, domain_num=6):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    lam = torch.rand(domain_num, batch_size).cuda()
    lam = F.softmax(lam * beta, dim=0)
    index_list = []
    mixed_x = 0
    for i in range(6):
        index_list.append(torch.randperm(batch_size).cuda())
        if self.flags.mix_from =='image':
            mixed_x += x[index_list[i], :] * lam[i].reshape(batch_size, 1, 1, 1)
        else:
            mixed_x += x[index_list[i], :] * lam[i].reshape(batch_size, 1)
    return mixed_x, index_list, y, lam

def mixup_criterion(pred, lam, index_list, y, domain_num=6):
    loss = 0
    for i in range(domain_num):
        loss += lam[i] * F.cross_entropy(pred, y[index_list[i]], reduction="none")
    return loss.mean()
```

The example steps are as follows:
```python
for (x,y) in iterDomainBatch:
    mixed_x, index_list, mixed_y, lam  = mixall(x, y)
    pred_y = model(mixed_x)
    loss = mixup_criterion(pred_y, lam, index_list, mixed_y)
    loss.backward()
    optimizer.step()
```
