from PIL import Image
from ultralytics import YOLO


# Load a pretrained YOLOv8n model
model1 = YOLO('seatbelt.pt')
model2 = YOLO('yawning.pt')
# Run inference on 'bus.jpg'
results1 = model1('vv.jpg')
results2 = model2('vv.jpg')

# Show the results
for r in results1:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image

for r in results2:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image


im.save('results.jpg')  # save image