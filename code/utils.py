import skimage
import skimage.io
import skimage.transform
import numpy as np
import os
from scipy.ndimage.interpolation import rotate
from random import random, shuffle
# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]

def randomly_rotate(image):
    rim = rotate(image, 90*int(4*np.random.uniform()))
    if np.random.random() > 0.5:
        rim = np.fliplr(rim)
    return rim    

def load_folder(dataset_path, max_no_img=256, crop_size=512):
    list_of_images=[]
    im_num=0
    for filename in os.listdir(dataset_path):
        if filename.endswith(".tiff") or filename.endswith(".png"):    
            #random rotation
            if im_num<max_no_img:
                im = load_image_fov(os.path.join(dataset_path, filename), crop_size)
                if im is not None and len(im.shape)==3:
                    
                    list_of_images.append(randomly_rotate(load_image_fov(os.path.join(dataset_path, filename), crop_size)))
                    im_num=im_num+1
    return  list_of_images       

def load_folder_random(dataset_path, max_no_img=200, crop_size=64):
    list_of_images=[]
    im_num=0
    imfiles = [[i] for i in os.listdir(dataset_path)]
    shuffle(imfiles)
    for filename in imfiles:
        if filename[0].endswith(".tiff") or filename[0].endswith(".png"):    
            #random rotation
            if im_num<max_no_img:
                im = load_image_fov(os.path.join(dataset_path, filename[0]), crop_size)
                if im is not None and len(im.shape)==3:
                    
                    list_of_images.append(randomly_rotate(im))
                    im_num=im_num+1
    return  list_of_images       


def load_image_fov(path, crop_size=512):
    # load image
    resized_img=None
    if os.path.getsize(path)>5000:
        img = skimage.io.imread(path)
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()
        # print "Original Image Shape: ", img.shape
        # we crop image from center
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        # resize to 224, 224
        resized_img = skimage.transform.resize(crop_img, (crop_size, crop_size))
    #except OSError as e:
        #print('Could not read:', path, ':', e, '- it\'s ok, skipping.')
        #pass
        
    return resized_img

def center_crop(image, width, height):
    offset_x = int((image.shape[0]-width)/2)
    offset_y = int((image.shape[1]-height)/2)
    cropped_image = image[offset_x:offset_x+width, offset_y:offset_y+height,:]
    return cropped_image


def random_crop(image,  split_size=512):
    epsi = 25
    image_width=image.shape[1] 
    image_height=image.shape[0]
    center_x = int(np.random.uniform(low=split_size+epsi, high=image_width-split_size-epsi))
    center_y = int(np.random.uniform(low=split_size+epsi, high=image_height-split_size-epsi))
    offset_x = int(center_x-split_size/2)
    offset_y = int(center_y-split_size/2)
    cropped_image = image[offset_x:offset_x+split_size, offset_y:offset_y+split_size]
    #cropped_image = image[offset_x:offset_x+split_size, offset_y:offset_y+split_size,:]
    return cropped_image


def load_image_crop_random(path, crop_size=512, re_size=512):
    # load image
    img = skimage.io.imread(path)
    
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    #rotate_img = randomly_rotate(img)
    #resized_img = center_crop(rotate_img, crop_size, crop_size)
    
    resized_img = random_crop(img, crop_size)#, crop_size)#skimage.transform.resize(crop_img, (crop_size, crop_size))
    #resized_img = skimage.transform.resize(crop_img, (re_size, re_size))
    #print(resized_img.shape)
    resized_img = resized_img / 255.0
    assert (0 <= resized_img).all() and (resized_img <= 1.0).all()
    
    return skimage.transform.resize(randomly_rotate(resized_img), (re_size, re_size))


def load_folder_random_crop(dataset_path, max_no_img=100, crop_size=512, re_size=256):
    list_of_images=[]
    im_num=0
    imfiles = [[i] for i in os.listdir(dataset_path)]
    shuffle(imfiles)
    for filename in imfiles:
        if filename[0].endswith(".tiff") or filename[0].endswith(".png"):    
            #random rotation
            if im_num<max_no_img:
                im = load_image_crop_random(os.path.join(dataset_path, filename[0]), crop_size, re_size)
                if im is not None and len(im.shape)==3:
                    
                    list_of_images.append(randomly_rotate(im))
                    im_num=im_num+1
    return  list_of_images       



def load_image_crop(path, crop_size=512, re_size=512):
    # load image
    img = skimage.io.imread(path)
    
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    #rotate_img = randomly_rotate(img)
    resized_img = center_crop(img, crop_size, crop_size)#skimage.transform.resize(crop_img, (crop_size, crop_size))
    resized_img = resized_img / 255.0
    assert (0 <= resized_img).all() and (resized_img <= 1.0).all()
    
    return skimage.transform.resize(resized_img, (re_size, re_size))




def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)
    
import tensorflow as tf    
def focal_loss(gamma=4., alpha=.75):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

def focal_loss_softmax(labels,logits,gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.softmax(logits,axis=-1) # [batch_size,num_classes]
   # labels=tf.one_hot(labels,depth=y_pred.shape[1])
    L=-labels*((1-y_pred)**gamma)*tf.log(y_pred)
    L=tf.reduce_sum(L,axis=1)
    return L    


if __name__ == "__main__":
    test()
