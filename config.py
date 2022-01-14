TARGET_LETTERS = 'KМ'
TARGET_NUMBERS = '134'
MODEL_FILE = 'model.pickle'
FONTS_FOLDER = 'fonts'
DATASET_FOLDER = 'dataset'
TEST_FILES = ['test1.png', 'test2.png']

IMAGE_SIZE = 28
SIZE = (IMAGE_SIZE, IMAGE_SIZE)
TARGET_ARRAY = [*TARGET_NUMBERS, *TARGET_LETTERS]
OUTPUTS_MAP = dict(zip(TARGET_ARRAY, range(len(TARGET_ARRAY))))
ITER_COUNT = 6000
INPUT_SIZE = IMAGE_SIZE ** 2
HIDDEN_SIZE = 128
OUTPUT_SIZE = len(OUTPUTS_MAP)
