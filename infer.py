import torch
from data import get_rubbish_classification_loader
from backbones import pyramidnet164

net = pyramidnet164(num_classes=16)
net.load_state_dict(torch.load("pyramidnet164-rubbish.pth"))
net.eval().requires_grad_(False).cuda()
test_loader = get_rubbish_classification_loader(mode="test", batch_size=1, shuffle=False)

with open("infer.txt", 'w') as f:
    for x, name in test_loader:
        x = x.cuda()
        y = torch.max(net(x), dim=1)[1].squeeze().item()

        string = f"{str(name[0])}  {str(y)}\n"
        f.write(string)


