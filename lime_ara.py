import os
import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
# print('Run using keras:', keras.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def show_image(explanation, EXPLAIN_LABEL, NUM_FEATURES, IMAGE_NR):
    temp, mask = explanation.get_image_and_mask(EXPLAIN_LABEL, positive_only=True, num_features=NUM_FEATURES, hide_rest=True)
    fig.add_subplot(2, 4, IMAGE_NR)
    plt.title(NUM_FEATURES)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.imshow(mark_boundaries(temp, mask)/255.0)


def transform_image(path_list):
    out = []
    for img_paths in path_list:
        img = image.load_img(img_paths, target_size=(150, 150), color_mode='rgb')
        # print(img.mode)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

import glob
img_paths = []
for filename in glob.glob('./dataset/test/*/*.jpg'):
    # print(filename)
    img_paths.append(filename)

images = transform_image(img_paths)

#load model
MODEL_PATH = './ARA_my_luty.h5'
from keras.models import load_model
model = load_model(MODEL_PATH)

preds = model.predict(images)

import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image

classnames=['01_TUMOR','02_STROMA','03_COMPLEX','04_LYMPHO','05_DEBRIS','06_MUCOSA','07_ADIPOSE','08_EMPTY']
explainer = lime_image.LimeImageExplainer()

from skimage.segmentation import mark_boundaries

file = open('explained_LIME.csv','w+')

for nr, img in enumerate(images):

    explanation = explainer.explain_instance(img, model.predict, top_labels=8, hide_color=0, num_samples=1000)
    # print(preds[nr])
    file.write('{}\nclassification result;{}\nprobability;{}\n\n'.format(img_paths[nr], ';'.join([str(classnames[elem]) for elem in explanation.top_labels]), ';'.join(['{}'.format(float(preds[nr][elem])) for elem in explanation.top_labels])))
    # print(preds[nr])
    # print(explanation.top_labels)

    for label in range(0,8):#0,8
        EXPLAIN_LABEL=explanation.top_labels[label]

        #show original image
        fig = plt.figure()
        fig.add_subplot(2, 4, 1)
        plt.title('original')
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        plt.imshow(image.load_img(img_paths[nr], target_size=(150, 150)))

        #show_image(explanation, EXPLAIN_LABEL, NUM_FEATURES, IMAGE_NR)
        show_image(explanation, EXPLAIN_LABEL, 1, 2)
        show_image(explanation, EXPLAIN_LABEL, 2, 3)
        show_image(explanation, EXPLAIN_LABEL, 5, 4)
        show_image(explanation, EXPLAIN_LABEL, 10, 5)
        show_image(explanation, EXPLAIN_LABEL, 20, 6)
        show_image(explanation, EXPLAIN_LABEL, 100, 7)
        show_image(explanation, EXPLAIN_LABEL, 1000, 8)

        plt.savefig('output/{}__{}_->_{}_{}.jpg'.format(img_paths[nr].split('/')[-1][:-4], img_paths[nr].split('/')[-2], classnames[explanation.top_labels[label]], label))
        # plt.show()
        plt.clf()
        plt.close()

file.close()
