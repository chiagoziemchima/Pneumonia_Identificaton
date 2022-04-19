
from __future__ import absolute_import

import tensorflow as tf
# noinspection PyUnresolvedReferences
from tensorflow.image import extract_patches
from tensorflow.keras.layers import Conv2D, Layer, Dense, Embedding

class patch_extract(Layer):
    def __init__(self, patch_size,**kwargs):
        super(patch_extract, self).__init__(**kwargs)
        self.patch_size = patch_size


    def get_config(self):

        return {
            'patch_size': self.patch_size
           }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, images):
        
        batch_size = tf.shape(images)[0]
        
        patches = extract_patches(images=images,
                                  sizes=(1, self.patch_size, self.patch_size, 1),
                                  strides=(1, self.patch_size, self.patch_size, 1),
                                  rates=(1, 1, 1, 1), padding='VALID',)
        # patches.shape = (num_sample, patch_num, patch_num, patch_size*channel)
        
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        patches = tf.reshape(patches, (batch_size, patch_num*patch_num, patch_dim))
        # patches.shape = (num_sample, patch_num*patch_num, patch_size*channel)
        
        return patches
    
class patch_embedding(Layer):
    def __init__(self, num_patch, embed_dim,**kwargs):
        super(patch_embedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.proj = Dense(self.embed_dim)
        self.pos_embed = Embedding(input_dim=num_patch, output_dim=self.embed_dim)

    def get_config(self):


        return {
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        embed = self.proj(patch) + self.pos_embed(pos)
        return embed

class patch_merging(tf.keras.layers.Layer):

    def __init__(self, num_patch, embed_dim, name='',**kwargs):
        super(patch_merging,self).__init__(**kwargs)
        
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        
        # A linear transform that doubles the channels 
        self.linear_trans = Dense(2*embed_dim, use_bias=False, name='{}_linear_trans'.format(name))

    def get_config(self):

        return {
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self, x):
        
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        
        assert (L == H * W), 'input feature has wrong size'
        assert (H % 2 == 0 and W % 2 == 0), '{}-by-{} patches received, they are not even.'.format(H, W)
        
        # Convert the patch sequence to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))
        
        # Downsample
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        
        # Convert to the patch squence
        x = tf.reshape(x, shape=(-1, (H//2)*(W//2), 4*C))
       
        # Linear transform
        x = self.linear_trans(x)

        return x

class patch_expanding(tf.keras.layers.Layer):

    def __init__(self, num_patch, embed_dim, upsample_rate, return_vector=True, name=''):
        super().__init__()
        
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        
        # Linear transformations that doubles the channels 
        self.linear_trans1 = Conv2D(upsample_rate*embed_dim, kernel_size=1, use_bias=False, name='{}_linear_trans1'.format(name))
        # 
        self.linear_trans2 = Conv2D(upsample_rate*embed_dim, kernel_size=1, use_bias=False, name='{}_linear_trans1'.format(name))
        self.prefix = name
        
    def call(self, x):
        
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        
        assert (L == H * W), 'input feature has wrong size'

        x = tf.reshape(x, (-1, H, W, C))
        
        x = self.linear_trans1(x)
        
        # rearange depth to number of patches
        x = tf.nn.depth_to_space(x, self.upsample_rate, data_format='NHWC', name='{}_d_to_space'.format(self.prefix))
        
        if self.return_vector:
            # Convert aligned patches to a patch sequence
            x = tf.reshape(x, (-1, L*self.upsample_rate*self.upsample_rate, C//2))

        return x