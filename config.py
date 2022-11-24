import torch
BATCH_SIZE = 4 # increase / decrease according to GPU memeory

NUM_EPOCHS = 10 # number of epochs to train for
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = '/Users/lukasrois/PycharmProjects/detecto/training_2022_WOE_base_20220517_cleaned_petonly'
# validation images and XML files directory
VALID_DIR = '/Users/lukasrois/PycharmProjects/detecto/test'
# classes: 0 index is reserved for background
CLASSES = ['background', 'pet_trans', 'pet_green', 'pet_blue']
NUM_CLASSES = 4
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = '/Users/lukasrois/PycharmProjects/detecto/outputs'
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs
