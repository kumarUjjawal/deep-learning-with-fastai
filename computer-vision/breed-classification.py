# Classifying Cats vs Dogs Breeds

from fastai.vision.all import *

path = untar_data(URLs.PETS)

files = get_image_files(path/"image")

pat = r'^(.*)_\d+.jpg'

dls = ImageDataLoaders.from_namme_re(path, files, pat, item_tfms=Resize(224), num_workers=0)

dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(460),
                                    batch_tfms=aug_transforms(224), num_workers)

learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.lr_find()

learn.fine_tune(2, 3e-3)

learn.show_results()

# check where the model made the wrost predictions

interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,10))
