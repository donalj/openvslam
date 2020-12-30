import torch, torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import visualize
import PIL
import cv2

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.cuda()
model.eval()

image = PIL.Image.open("image.png")
image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()

# plt.imshow(image)
# image = transforms.ToPILImage()(image)
# transform = transforms.Compose([transforms.ToTensor()])
# image = transform(image)
prediction = model([image_tensor])
# print(r)
# visualize.display_instances(image, r['boxes'], r['masks'], r['labels'],
#                             class_names, r['scores'])
img_cv = cv2.imread('image.png', cv2.COLOR_BGR2RGB)

for i in range(len(prediction[0]['masks'])):
    # iterate over masks
    mask = prediction[0]['masks'][i, 0]
    mask = mask.mul(255).byte().cpu().numpy()
    _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_cv, contours, -1, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imwrite('output.png', img_cv)
