import torch, torchvision

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
script_model = torch.jit.script(model)
script_model.save("./models/maskrcnn_resnet50_fpn.pt")

model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model.eval()
script_model = torch.jit.script(model)
script_model.save("./models/deeplabv3_resnet50.pt")

model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()
script_model = torch.jit.script(model)
script_model.save("./models/deeplabv3_resnet101.pt")

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()
script_model = torch.jit.script(model)
script_model.save("./models/keypointrcnn_resnet50_fpn.pt")
