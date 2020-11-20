from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import get_object_or_404, render
from django.template import loader
from django.http import HttpResponse
from .forms import *
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from keras.models import load_model
# Create your views here.


from scipy import ndimage
from skimage.measure import label, regionprops
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import os
import sys



from keras.utils import to_categorical
from keras import layers, initializers, regularizers, constraints
from keras.layers import Layer, InputSpec

from keras import backend as K
import keras

from skimage.filters import threshold_triangle
from skimage.morphology import remove_small_objects
from skimage.measure import marching_cubes_lewiner
from keras.models import load_model

global graph, model, sess
sess = tf.Session()
set_session(sess)
graph = tf.get_default_graph()
sess.run(tf.global_variables_initializer())


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


@csrf_exempt
def deface(request):
    context = dict()
    context['title'] = 'title'
    context['events'] = list()

    

    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
        upload_file = os.path.join('tmp', request.FILES['fileToDeface'].name)
        with open(upload_file, 'wb+') as destination:
            for chunk in request.FILES['fileToDeface'].chunks():
                destination.write(chunk)
        Deidentification_image(where=(1, 1, 1,1), Type='wipe', list_test_image=[os.path.abspath(upload_file)], model=model)
        
        result_file = os.path.join('tmp', 'Wipe_{}'.format(request.FILES['fileToDeface'].name))
        with open(result_file, 'rb') as file:
            response = HttpResponse(file, content_type='application/javascript')
            response['Content-Encoding'] = 'gzip'
            response['Content-Disposition'] = 'attachment; filename=%s' % os.path.basename(result_file)
        return response
        
    return

config = dict()
config["img_channel"] = 1
config["num_multilabel"] = 5 # the number of label (channel last)
config["noise"] = 0.1
config["batch_size"] = 1 # 3D segmentation learning needs too large GPU memory to increase batch size. # this script is optimized for single batch size
config["resizing"] = True #True -> resize input image for learning. if you don't have enough GPU memory.
config["input_shape"] = [128, 128, 128, 1] # smaller GPU memory smaller image size


def dice_score(y_true, y_pred):
    smooth = 1.
    label_length = y_pred.get_shape().as_list()[-1]  # the number of label (channel last)

    loss = 0
    for num_labels in range(label_length):
        y_true_f = K.flatten(y_true[..., num_labels])
        y_pred_f = K.flatten(y_pred[..., num_labels])
        intersection = K.sum(y_true_f * y_pred_f)
        loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return loss / label_length


def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)


def resize(data, img_dep=config["input_shape"][0], img_cols=config["input_shape"][1], img_rows=config["input_shape"][2]):
    resize_factor = (img_dep/data.shape[0], img_cols/data.shape[1], img_rows/data.shape[2])
    data = ndimage.zoom(data, resize_factor, order=0, mode='constant', cval=0.0)
    return data


class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """

    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# %%
#
model = load_model('/home/shmoon/PycharmProjects/core_service/model_2.h5', custom_objects={
        'InstanceNormalization': InstanceNormalization,
        'dice_loss': dice_loss,
        'dice_score': dice_score
    })

# 5D tensor (batch,img_dep,img_cols,img_rows,img_channel)
def load_batch(x_list,y_list=0, batch_size=1):
    
    image = sitk.GetArrayFromImage(sitk.ReadImage(x_list)).astype('float32')
    if config["resizing"] == True : 
        image = resize(image)
        img_shape = image.shape
    else:
        img_shape = image.shape

    image = np.reshape(image,(config["batch_size"] ,img_shape[0], img_shape[1], img_shape[2] , config["img_channel"])) # batch, z ,y, x , ch
    n_image = (image-np.min(image))/(np.max(image)-np.min(image))
  
    label=0
    if y_list!=0:
        labels = sitk.GetArrayFromImage(sitk.ReadImage(y_list)).astype('float32')
        if config["resizing"] == True : 
            labels = resize(labels)
            lb_shape = labels.shape
        else:
            lb_shape = labels.shape
            

        onehot = to_categorical(labels)     
        label = np.reshape(onehot,(config["batch_size"] ,lb_shape[0], lb_shape[1], lb_shape[2], config["num_multilabel"] ))

    
     
    return n_image,label
       

    


def onehot2label(onehot_array):
    onehot_array = np.argmax(onehot_array, axis=-1)
    label = onehot_array[..., np.newaxis]

    return label


def bounding_box(results):

    boxes = list()
    for ch in range(results.shape[-1]):  # except 0 label (blanck)
        if ch == 0 or ch == 2: # eyes, ears
            
            result = np.round(results[...,ch])
            lb=label(result,connectivity=1)
            
            if np.max(lb) > 2:
                region_list = [region.area for region in regionprops(lb)]       
                lb=remove_small_objects(lb, min_size=np.max(region_list)*0.3)
            
            if len(regionprops(lb))!=2 :
                raise print('\nROI detection failed')

            for region in regionprops(lb):
                 boxes.append(list(region.bbox))
        
        
        if ch==1 or ch==3: # nose
           
            result = np.round(results[...,ch])
            lb=label(result,connectivity=1)

            if np.max(lb) > 1:
                region_list = [region.area for region in regionprops(lb)]       
                lb=remove_small_objects(lb, min_size=np.max(region_list)*0.3)
            
            if len(regionprops(lb))!=1 :
                raise print('\nROI detection failed')

            for region in regionprops(lb):
                 boxes.append(list(region.bbox))
    
    return boxes

def box_blur(im_array,  box, Type, wth=1):
    
    # increase or decrease the size of the box by 'wth' times
    if wth != 1 :
        for c in range(3):
            mean_= (box[c]+box[c+3])/2
            box[c]=int(np.round(mean_-wth*(mean_-box[c])))
            box[c+3]=int(np.round(wth*(box[c+3]-mean_)+mean_))
            if box[c] < 0: box[c]=0
            if box[c+3] > im_array.shape[2-c]: box[c+3] = im_array.shape[2-c]   #order : im_array-> x,y,z / box-> z,y,x

 
    
    #voxel coordinates must be 'int'
    box_z1 = box[0]
    box_y1 = box[1]
    box_x1 = box[2]
    box_z2 = box[3]
    box_y2 = box[4]
    box_x2 = box[5]
    
    
    if Type.lower() =='blur':
        blurr_array = ndimage.median_filter(im_array[box_x1:box_x2,box_y1:box_y2,box_z1:box_z2],size=5) #size = filter size
        im_array[box_x1:box_x2,box_y1:box_y2,box_z1:box_z2] = blurr_array
    elif Type.lower() =='wipe':
        
        im_array[box_x1:box_x2,box_y1:box_y2,box_z1:box_z2] = 0
    else:
        raise print("type option error : select 'Blur' or 'Wipe'")
    
    return im_array

def surface_blur(im_array, edge_img, box, Type, wth ,dep):
    
    
    # increase or decrease the size of the box by 'wth' times
    if wth != 1 :
        for c in range(3):
            mean_= (box[c]+box[c+3])/2
            box[c]=int(np.round(mean_-wth*(mean_-box[c])))
            box[c+3]=int(np.round(wth*(box[c+3]-mean_)+mean_))
            if box[c] < 0: box[c]=0
            if box[c+3] > im_array.shape[2-c]: box[c+3] = im_array.shape[2-c]   #order : im_array-> x,y,z / box-> z,y,x

    
    
    #voxel coordinates must be 'int'
    box_z1 = box[0]
    box_y1 = box[1]
    box_x1 = box[2]
    box_z2 = box[3]
    box_y2 = box[4]
    box_x2 = box[5]
    
    mini_array = im_array[box_x1:box_x2,box_y1:box_y2,box_z1:box_z2]            
    mini_edge = edge_img[box_x1:box_x2,box_y1:box_y2,box_z1:box_z2]
    processing_area = np.zeros_like(mini_array)

    
    # dep  =eyes=2~3 ears 1~2
    if Type.lower() =='blur':
        where_true=np.where(mini_edge==True)
        
        for i in range(len(where_true[0])):
            x=where_true[0][i]
            y=where_true[1][i]
            z=where_true[2][i]
            processing_area[x-dep:x+dep,y-dep:y+dep,z-dep:z+dep] = 1
            
        
        
        threshold = np.max(ndimage.gaussian_filter(mini_array[processing_area==1],sigma=3))
        mini_array[processing_area==1] = threshold
 
        
    
    elif Type.lower() =='wipe':
        where_true=np.where(mini_edge==True)

        for i in range(len(where_true[0])):
            x=where_true[0][i]
            y=where_true[1][i]
            z=where_true[2][i]
            mini_array[x-dep:x+dep,y-dep:y+dep,z-dep:z+dep]=0

    
    else:
        raise print("type option error : select 'Blur' or 'Wipe'")
    
    
    im_array[box_x1:box_x2,box_y1:box_y2,box_z1:box_z2] = mini_array 
    
    return im_array


def outer_contour_3D(image,zoom=1):
    #sort in standard size
    resize_factor = (128/image.shape[0], 128/image.shape[1], 128/image.shape[2])
    ima = ndimage.zoom(image,resize_factor,order=0, mode='constant', cval=0.0)
    
    # make binary cast
    thresh = threshold_triangle(ima)
    imageg=ndimage.median_filter(ima,size=3)
    binary_image = imageg > thresh
    for s in range(ima.shape[0]) :
        binary_image[s,:,:]=ndimage.morphology.binary_fill_holes(binary_image[s,:,:])
    for s in range(ima.shape[1]) :
        binary_image[:,s,:]=ndimage.morphology.binary_fill_holes(binary_image[:,s,:])
    for s in range(ima.shape[2]) :
        binary_image[:,:,s]=ndimage.morphology.binary_fill_holes(binary_image[:,:,s])
    
    # draw outer contour
    verts, faces, norm, val = marching_cubes_lewiner(binary_image,0)
    vint = np.round(verts).astype('int')
    contour = np.zeros_like(binary_image)
    for s in vint:
        contour[s[0],s[1],s[2]]=1
    
    # shrink contour image cuz of the gaussian_filter we used earlier.
    if zoom !=1 :
        c_shape = contour.shape
        zoom_ = ndimage.zoom(contour,zoom,order=0, mode='constant', cval=0.0)
        zoom_shape = zoom_.shape
        npad = ( (int(np.ceil((c_shape[0]-zoom_shape[0])/2)),int((c_shape[0]-zoom_shape[0])/2)), 
                (int(np.ceil((c_shape[1]-zoom_shape[1])/2)),int((c_shape[1]-zoom_shape[1])/2)),
                (int(np.ceil((c_shape[2]-zoom_shape[2])/2)),int((c_shape[2]-zoom_shape[2])/2)) )

        contour_3D = np.pad(zoom_,npad,'constant',constant_values = (0))
    elif zoom==1 : 
        contour_3D = contour
    
    #Revert to original size
    get_back = (image.shape[0]/128, image.shape[1]/128, image.shape[2]/128)
    contour_3D = ndimage.zoom(contour_3D,get_back,order=0, mode='constant', cval=0.0)
    
    return contour_3D


def Deidentification_image(where, Type, list_test_image, model=model):
    '''
    where : list or tuple. Each position stands for eyes nose ears (eyes, nose, ears) 
            If the corresponding position is 1, de-identification process.
    
    Type : Image processing options. 'Wipe' , 'Blurr'
    
    list_of_image : Test set(labled or unlabled) data path. 
    model : Predictive model to be applied.

    '''
    # if typing another word, error raise. 
    options=['blur','wipe']
    if Type.lower() not in options: 
        raise print("type option error : select 'Blur' or 'Wipe'") 
    
    
    
    for i in range(len(list_test_image)): # load image of i th : 0 ~
        
        raw_img = nib.load(list_test_image[i]) # get affine and header of original image file.
        array_img = raw_img.get_fdata() # image array
        original_shape = array_img.shape #  (x,y,z)
        thresh = threshold_triangle(array_img)
        
        #load prediction label 
        image, label = load_batch(list_test_image[i]) #z, y, x 

        with graph.as_default(): 
            set_session(sess)
            results = model.predict(image)

        results = np.round(results)
        
        
        #preprocessing: Size recovery and transform onehot to labels number 
        if config["resizing"]==True :
            results = onehot2label(results)
            results = np.reshape(results,config["input_shape"][0:3]) # prediction results (batch size, dep, col ,row, ch) -> (dep, col ,row)
            results = resize(results,
                                 img_dep=original_shape[2],
                                 img_cols=original_shape[1],
                                 img_rows=original_shape[0]) 
            results = to_categorical(results)# except 0 label (blanck)

        else:
            results=results[0,...] # Only if batch size==1

        
        #search center by clustering
        boxes = bounding_box(results[...,1:])    
        
       
        
        
        # 
        if where[1] == 1: # nose
            box = boxes[2]
            array_img = box_blur(array_img, box, Type='wipe', wth=1.33)
        
        ## make outer contour for mini array.
        edge_img = outer_contour_3D(array_img,zoom=1)
        
        
        if where[0] == 1: # eyes
                                   
            box = boxes[0] # eye
            array_img = surface_blur(array_img, edge_img, box, Type='blur', wth=1.5, dep=3)
            
            box = boxes[1] # eye
            array_img = surface_blur(array_img, edge_img, box, Type='blur', wth=1.5, dep=3)               
        
        
            
        if where[2] == 1: # ears
            '''
            In order not to see the outline of the ear due to external noise,
            fill the area of the ear with similar noise
            '''
            ear_results = results[...,3]
            border = box_blur(np.ones(array_img.shape),boxes[3],Type='wipe')
            border = box_blur(border,boxes[4],Type='wipe')
            border = 1-border
            ear_results = border*ear_results.T
            
            noise = np.random.rand(*original_shape)*thresh*0.8 
            array_img[ear_results == 1] = noise[ear_results == 1]

        if where[3] : # mouth
            mouth_results = results[...,4] 
            border = box_blur(np.ones(array_img.shape),boxes[5],Type='wipe') 
            border = 1-border
            if where[1] == False: # If you want to preserve the nose
                    border = box_blur(border,boxes[2],Type='wipe', wth=1.5)
            mouth_results = border*mouth_results.T
            
            threshold = np.max(ndimage.gaussian_filter(array_img[mouth_results==1],sigma=3))
            array_img[mouth_results==1] = threshold

        array_img=np.round(array_img)
        array_img=np.array(array_img,dtype='int32')
        

        if Type.lower() == 'blur':
            nib.save(nib.Nifti1Image(array_img, raw_img.affine, raw_img.header),
                     os.path.join('tmp',
                                  'Blurred_{}'.format(os.path.basename(list_test_image[i]))))
            #r.append('Blurred_{}'.format(os.path.basename(list_test_image[i])))
            
        elif Type.lower() == 'wipe':
            nib.save(nib.Nifti1Image(array_img, raw_img.affine, raw_img.header),
                     os.path.join('tmp',
                                  'Wipe_{}'.format(os.path.basename(list_test_image[i]))))
            #r.append('Wipe_{}'.format(os.path.basename(list_test_image[i])))

    #return r


