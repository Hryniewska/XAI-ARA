import os
import keras
from keras.preprocessing import image
# from keras.applications.imagenet_utils import decode_predictions
# from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
print('Run using keras:', keras.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def show_image(explanation, EXPLAIN_LABEL, NUM_FEATURES, IMAGE_NR):
    temp, mask = explanation.get_image_and_mask(EXPLAIN_LABEL, positive_only=True, num_features=NUM_FEATURES, hide_rest=True)
    fig.add_subplot(2, 4, IMAGE_NR)
    plt.title(NUM_FEATURES)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.imshow(mark_boundaries(temp, mask))


def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(150, 150))
        print(img.mode)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

IMAGE_NAME='01_TUMOR/5ADB_CRC-Prim-HE-04_029.jpg_Row_451_Col_1.jpg'
img_path = [os.path.join('./dataset/test', IMAGE_NAME)]
images = transform_img_fn(img_path)

fig = plt.figure()
fig.add_subplot(2, 4, 1)
plt.title('original')
plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
plt.imshow(image.load_img(img_path[0], target_size=(150, 150)))

#import test_model
MODEL_PATH = './ARA_my.h5'
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

explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(images[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)

from skimage.segmentation import mark_boundaries

print(preds)
print(explanation.top_labels)
print(explanation.top_labels[1])

EXPLAIN_LABEL=explanation.top_labels[1]

#show_image(explanation, EXPLAIN_LABEL, NUM_FEATURES, IMAGE_NR)
show_image(explanation, EXPLAIN_LABEL, 1, 2)
show_image(explanation, EXPLAIN_LABEL, 2, 3)
show_image(explanation, EXPLAIN_LABEL, 5, 4)
show_image(explanation, EXPLAIN_LABEL, 10, 5)
show_image(explanation, EXPLAIN_LABEL, 20, 6)
show_image(explanation, EXPLAIN_LABEL, 100, 7)
show_image(explanation, EXPLAIN_LABEL, 1000, 8)

plt.savefig('output/' + IMAGE_NAME.replace('/', '_'))
plt.show()

import keras
#
# MODEL_PATH = './ARA_my_4.h5'
# from keras.models import load_model
# model = load_model(MODEL_PATH)
# import shap
# import numpy as np
#
# # select a set of background examples to take an expectation over
# background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
#
# # explain predictions of the model on four images
# e = shap.DeepExplainer(model, background)
# # ...or pass tensors directly
# # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
# shap_values = e.shap_values(x_test[1:5])
#
# # plot the feature attributions
# shap.image_plot(shap_values, -x_test[1:5])
