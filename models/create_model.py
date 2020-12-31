import torch
import torchvision
import os.path


if (not os.path.isfile("./models/maskrcnn_resnet50_fpn.pt")) :
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    script_model = torch.jit.script(model)
    script_model.save("./models/maskrcnn_resnet50_fpn.pt")

if (not os.path.isfile("./models/deeplabv3_resnet50.pt")) :
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.eval()
    script_model = torch.jit.script(model)
    script_model.save("./models/deeplabv3_resnet50.pt")

if (not os.path.isfile("./models/deeplabv3_resnet101.pt")) :
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    script_model = torch.jit.script(model)
    script_model.save("./models/deeplabv3_resnet101.pt")

if (not os.path.isfile("./models/keypointrcnn_resnet50_fpn.pt")) :
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    script_model = torch.jit.script(model)
    script_model.save("./models/keypointrcnn_resnet50_fpn.pt")

if (not os.path.isfile("./models/MiDaS.pt")) :
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.eval()
    script_model = torch.jit.script(model)
    script_model.save("./models/MiDaS.pt")
