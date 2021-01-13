# Create a Learner to classify cats vs dogs using Resnet34 architecture. We'll use Oxford-IIIT Pet dataset for this # the dataset contains images of cats and dogs of 37 breeds.

from fastai.vision.all import *

path = untar_data(URLs.PETS)

files = get_image_files(path/"image")

def label_func(f):
    return f[0].isupper()

# data loaders
dls = ImageDataLoader.from_name_func(path, files, label_func, item_tfms=Resize(224), num_workers=0)

learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fine_tune(1)

# predict
learn.predict(files[0])

learn.show_results()
