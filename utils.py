from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil


def load_data(data_path='./pine_water_cc.mat', validation_rate=0.25):
    if data_path == './pine_water_cc.mat':
        data = loadmat(data_path)
        y_train, y_test = data['value_train'], data['value_test']
        print('Value train shape: ', y_train.shape, 'Value test shape', y_test.shape)
        y_max_value, y_min_value = data['value_max'], data['value_min']
        x_train, x_test = data['DL_train'], data['DL_test']
    elif data_path == './N_100_leaf_cc.mat':
        data = loadmat(data_path)
        y_train, y_test = data['y_train'], data['y_test']
        x_train, x_test = data['x_train'], data['x_test']
        y_max_value, y_min_value = data['max_y'], data['min_y']
        x_train = np.expand_dims(x_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
        x_validation, y_validation = x_test, y_test
        return x_train, x_test, x_validation, y_train, y_test, y_validation, y_max_value, y_min_value
    else:
        data = loadmat(data_path)
        y_train, y_test = data['y_train'], data['y_test']
        x_train, x_test = data['x_train'], data['x_test']
        y_max_value, y_min_value = data['max_y'], data['min_y']
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)
    print('SG17 DATA train shape: ', x_train.shape, 'SG17 DATA test shape', x_test.shape)

    print('Mini value: %s, Max value %s.' % (y_min_value, y_max_value))

    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_rate,
                                                                    random_state=8)

    return x_train, x_test, x_validation, y_train, y_test, y_validation, y_max_value, y_min_value


def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    :param dir_name: 文件夹
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print('[Info] 文件夹 "%s" 存在, 删除文件夹.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('[Info] 文件夹 "%s" 不存在, 创建文件夹.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False


class Config:
    def __init__(self):
        # 数据有关的参数
        self.validation_rate = 0.2
        # 训练有关参数
        self.train_epoch = 20000
        self.batch_size = 20
        # 是否训练的参数
        self.train_cnn = True
        self.train_ms_cnn = True
        self.train_ms_sc_cnn = True
        # 是否评估参数
        self.evaluate_cnn = True
        self.evaluate_ms_cnn = True
        self.evaluate_ms_sc_cnn = True
        # 要评估的保存好的模型列表
        self.evaluate_cnn_name_list = []
        self.evaluate_ms_cnn_name_list = []
        self.evaluate_ms_sc_cnn_name_list = []

        # 存储训练出的模型和图片的文件夹
        self.img_dir = './pictures0331'
        self.checkpoint_dir = './check_points0331'

        # 数据集选择
        self.data_set = './dataset_preprocess/corn/corn_mositure.mat'

    def show_yourself(self, to_text_file=None):
        line_width = 36
        content = '\n'
        # create line
        line_text = 'Data Parameters'
        line = '='*((line_width-len(line_text))//2) + line_text + '='*((line_width-len(line_text))//2)
        line.ljust(line_width, '=')
        content += line + '\n'
        content += 'Validation Rate: ' + str(self.validation_rate) + '\n'
        # create line
        line_text = 'Training Parameters'
        line = '=' * ((line_width - len(line_text)) // 2) + line_text + '=' * ((line_width - len(line_text)) // 2)
        line.ljust(line_width, '=')
        content += line + '\n'
        content += 'Train CNN: ' + str(self.train_cnn) + '\n'
        content += 'Train Ms CNN: ' + str(self.train_ms_cnn) + '\n'
        content += 'Train Ms Sc CNN: ' + str(self.train_ms_sc_cnn) + '\n'
        # create line
        line_text = 'Evaluate Parameters'
        line = '=' * ((line_width - len(line_text)) // 2) + line_text + '=' * ((line_width - len(line_text)) // 2)
        line.ljust(line_width, '=')
        content += line + '\n'
        content += 'Train Epoch: ' + str(self.train_epoch) + '\n'
        content += 'Train Batch Size: ' + str(self.batch_size) + '\n'

        content += 'Evaluate CNN: ' + str(self.evaluate_cnn) + '\n'
        if len(self.evaluate_cnn_name_list) >=1:
            content += 'Saved CNNs to Evaluate:\n'
            for models in self.evaluate_cnn_name_list:
                content += models + '\n'

        content += 'Evaluate Ms CNN: ' + str(self.evaluate_ms_cnn) + '\n'
        if len(self.evaluate_ms_cnn_name_list) >= 1:
            content += 'Saved Ms CNNs to Evaluate:\n'
            for models in self.evaluate_ms_cnn_name_list:
                content += models + '\n'

        content += 'Evaluate Ms Sc CNN: ' + str(self.evaluate_ms_cnn) + '\n'
        if len(self.evaluate_ms_sc_cnn_name_list) >= 1:
            content += 'Saved Ms Sc CNNs to Evaluate:\n'
            for models in self.evaluate_ms_sc_cnn_name_list:
                content += models + '\n'

        # create line
        line_text = 'Saving Dir'
        line = '=' * ((line_width - len(line_text)) // 2) + line_text + '=' * ((line_width - len(line_text)) // 2)
        line.ljust(line_width, '=')
        content += line + '\n'
        content += 'Image Dir: ' + str(self.img_dir) + '\n'
        content += 'Check Point Dir: ' + str(self.img_dir) + '\n'
        print(content)
        if to_text_file:
            with open(to_text_file, 'w') as f:
                f.write(content)
        return content


if __name__ == '__main__':
    config = Config()
    config.show_yourself(to_text_file='name.txt')
    x_train, x_test, x_validation, y_train, y_test, y_validation, y_max_value, y_min_value = \
        load_data(data_path='./yaowan_calibrate.mat', validation_rate=0.25)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, x_validation.shape, y_validation.shape,
          y_max_value, y_min_value)
