import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical




class FGenerator(tf.keras.utils.Sequence):

    SEGMENTATION='segmentation'
    CLASSIFICATION='classification'
    MULTI_LABEL='multi_label'

    """ generates random segmentation/label/multi-class data

    Args:
        
        shape<int>: batch shape (batch_size,height,width,nb_bands)
        nb_cats<int>:
            * number of categories
            * does not include zero (which treated as no data value)
        nb_batches<int>: number of batches per epoch
        min_blobs<int>: min number of blobs
        max_blobs<int>: max number of blobs
        min_size<int>: min height/width of blobs
        max_size<int>: max height/width of blobs
        noise<float>: amount of noise in input data
        background_noise<float>: amount of background noise
        init_cat<int>: init_cat
        onehot<bool>: if true return categorical targets 
        target_type<str>:
            * segmentation: returns images with blocks of data
            * classification returns the class value of the most frequent category
            * multi_label returns the class value of all the blocks

    Batch:

        * target data is blocks of category values
        * input data is noise (on a random band) around the target category value
    """
    def __init__(self,
            shape,
            nb_cats=3,
            nb_batches=3,
            min_blobs=2,
            max_blobs=12,
            min_size=3,
            max_size=40,
            noise=0.75,
            background_noise=1.0,
            init_cat=0,
            onehot=True,
            target_type=SEGMENTATION,
            **handler_kwargs):
        self.shape=list(shape)
        self.nb_cats=nb_cats
        self.batch_size=shape[0]
        self.nb_batches=nb_batches
        self.min_blobs=min_blobs
        self.max_blobs=max_blobs
        self.min_size=min_size
        self.max_size=max_size
        self.noise=noise
        self.background_noise=background_noise
        self.init_cat=init_cat
        self.onehot=onehot
        self.target_type=target_type
        self.values=list(range(1,nb_cats+1))

    

    #
    # Sequence Interface
    #
    def __len__(self):
        """ number of batches """
        return self.nb_batches
    
    
    def __getitem__(self,batch_index):
        """ return input-target batch """
        inpts, targs=self.blob_batch()
        if self.onehot:
            targs=to_categorical(targs,num_classes=self.nb_cats+1)
        return inpts, targs
    

    def on_epoch_end(self):
        """ on-epoch-end callback """
        print('boom')



    #
    # Public
    #    
    def blob_images(self,inpt=None,targ=None):
        if inpt is None:
            inpt=np.zeros(self.shape[1:])
            inpt+=self.background_noise*np.random.random(self.shape[1:])
        if targ is None:
            targ=np.full(self.shape[1:-1],self.init_cat).astype(np.float16)
        nb_blobs=np.random.randint(self.min_blobs,self.max_blobs)
        for _ in range(nb_blobs):
            band=np.random.randint(0,self.shape[-1])
            sy, sx=self._random_shape()
            y, x=self._random_position(sy,sx)
            iblob, tblob=self._build_blob(sy,sx)
            inpt[y:y+sy,x:x+sx,band]=iblob
            targ[y:y+sy,x:x+sx]=tblob
        return inpt, targ


    def blob_batch(self):
        inpts=np.zeros(self.shape)+self.background_noise*np.random.random(self.shape)
        targs=np.full(self.shape[:-1],self.init_cat).astype(np.float16)
        for b in range(self.shape[0]):
            iblob,tblob=self.blob_images(inpts[b],targs[b])
            inpts[b]=iblob
            targs[b]=tblob
        if self.target_type==FGenerator.CLASSIFICATION:
            targs=[ self._get_classification(t) for t in targs ]
        elif self.target_type==FGenerator.MULTI_LABEL:
            targs=[ self._get_labels(t) for t in targs ]
        return inpts, targs


    #
    # INTERNAL
    #
    def _build_blob(self,sy,sx):
        v=np.random.choice(self.values)
        targ=np.full((1,sy,sx),v)
        inpt=targ+self.noise*np.random.randn(1,sy,sx)
        return inpt, targ


    def _random_shape(self):
        sy=np.random.randint(self.min_size,self.max_size+1)
        sx=np.random.randint(self.min_size,self.max_size+1)
        return sy, sx

    
    def _random_position(self,sy,sx):
        y=np.random.randint(0,self.shape[-3]-sy)
        x=np.random.randint(0,self.shape[-2]-sx)
        return y, x


    def _get_classification(self,targ):
        values,counts=np.unique(targ[targ!=0],return_counts=True)
        return values[np.argmax(counts)]


    def _get_labels(self,targ):
        return list(np.unique(targ[targ!=0]))


