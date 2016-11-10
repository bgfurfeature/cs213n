# png pixel extractor
import numpy as np
import cPickle as pickle
from PIL import Image


class Image_process(object):
    """
      get pixels from .png,.jpg files
    """

    def __init__(self, file_path=None, num_pic=1):
        self.file = file_path
        self.num_pic = num_pic
        self.pixel_dict = {}
        self.pixel_array = []
        self.height = self.weight = 32
        self.channel = 3

    # get pixels of a image
    def pixels(self):
        for i in range(1, self.num_pic + 1):
            file_name = self.file + str(i) + ".png"
            image_object = Image.open(file_name, mode='r')
            r, g, b = image_object.split()
            image_arr = [np.asanyarray(channel, dtype=np.uint8).reshape([1, self.height * self.weight]) for channel in
                         image_object.split()]
            # r_arr = np.asanyarray(r, dtype=np.uint8)
            # g_arr = np.asanyarray(g, dtype=np.uint8)
            # b_arr = np.asanyarray(b, dtype=np.uint8)
            self.pixel_array.append(image_arr)

    # save to pickle file
    def save_as_pickle(self, file_name=None):
        if file_name is not None:
            path = self.file + str(file_name)
        else:
            print 'file_name should not be None!!'
            return
        data_format = np.asanyarray(self.pixel_array, dtype=np.uint8).reshape(
            (self.num_pic, self.height * self.weight * self.channel))
        self.pixel_dict = {'data': data_format}
        print self.pixel_dict['data'].shape
        f = file(path, 'wb')
        pickle.dump(self.pixel_dict, f, True)
        f.close()
        # f2 = file(path, 'rb')
        # res = pickle.load(f2)
        # f2.close()


if __name__ == '__main__':
    image = Image_process(file_path="H:/MachineLearning/DataSet/kaggle.com/cifar-10/test_2/test/",
                          num_pic=300000)
    image.pixels()
    image.save_as_pickle("test_data_batch")
    print 'finished!!'
