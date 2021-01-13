# using data block api to use single-label classification
from fastai.vision.all import *

pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                    get_items=get_image_files,
                    splitter=RandomSplitter,
                    get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                    item_tfms=Resize(460),
                    batch_tfms=aug_transforms(size=224))

dls = pets.dataloaders(untar_data(URLs.PETS)/"images")

learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fine_tune(1)
