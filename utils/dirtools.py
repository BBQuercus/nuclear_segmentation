import random
import csbdeep.utils

def train_valid_split(x_list, y_list, valid_split=0.25):
    '''Splits two lists (images and masks) into random training and validation sets.
    '''
    assert all(csbdeep.utils.Path(x).name==csbdeep.utils.Path(y).name for x, y in zip(x_list, y_list))

    # Random shuffle
    combined = list(zip(x_list, y_list))
    random.shuffle(combined)
    x_list, y_list = zip(*combined)

    # Split into train / valid
    split_len = round(len(x_list) * valid_split)
    x_valid, x_train = x_list[:split_len], x_list[split_len:]
    y_valid, y_train = y_list[:split_len], y_list[split_len:]

    print(f'Total images: {len(x_list)}')
    print(f'– training images: {len(x_train)}')
    print(f'– validation images: {len(x_valid)}')

    return x_train, x_valid, y_train, y_valid