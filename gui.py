from tkinter import *    # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
import torch

from model import create_model
import os
root = Tk()

# Load model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=4).to(device)
model.load_state_dict(torch.load('/Users/lukasrois/PycharmProjects/detecto/model_file.pth'))
model.eval()


#Resize window
root.geometry("420x80")

# Define Function for the buttons
def btn1():
    root.destroy()


def btn2():
    exit()

# Create a window describing the program and add a button to start the program


T = Text(root, height=5, width=52)
# Create label
l = Label(root, text="Welcome to the Recycle Bot!")
c = Label(root, text="Select a Image for the Recycle Bot to predict")

l.config(font=("Courier", 25))
button1 = Button(root, text="Press to get Started", command=btn1)
l.pack()
c.pack()
button1.pack()
root.mainloop()


while True:
    # Open file chooser and save file
    filename = askopenfilename()

    jpg_files = [filename]




    test_images = jpg_files
    print(f"Test instances: {len(test_images)}")
    # classes: 0 index is reserved for background
    CLASSES = ['background', 'pet_trans', 'pet_green', 'pet_blue']
    # define the detection threshold...I love the way there to graduation
    # ... any detection having score below this will be discarded
    detection_threshold = 0.8
    print(test_images[0])

    for i in range(len(test_images)):
        # get the image file name for saving output later on
        image_name = test_images[i].split('/')[-1].split('.')[0]
        image = cv2.imread(os.path.join(filename, test_images[i]))

        # Pre-proccess image

        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float64)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float)
        # add batch dimension
        image = torch.unsqueeze(image, 0)

        # Turn off gradients and predict the image labels and boxes with the model

        with torch.no_grad():
            outputs = model(image)

        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(orig_image,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 0, 255), 2)
                cv2.putText(orig_image, pred_classes[j],
                            (int(box[0]), int(box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2, lineType=cv2.LINE_AA)
            cv2.imshow('Prediction', orig_image)
            cv2.waitKey(0)
            cv2.imwrite(f"../test_predictions/{image_name}.jpg", orig_image, )
        print(f"Image {i + 1} done...")
        print('-' * 50)
    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()
