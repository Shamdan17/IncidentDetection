import tensorflow
from tensorflow import keras
from keras.layers import Dense, Flatten, Permute
from keras import Sequential
import keras.backend as kb
kb.set_image_data_format('channels_first')
import resnet
from util import (
    get_place_to_index_mapping,
    get_incident_to_index_mapping,
    get_index_to_incident_mapping,
    get_index_to_place_mapping
)


place_to_idx = get_place_to_index_mapping()
incident_to_idx = get_incident_to_index_mapping()

index_to_incident_mapping = get_index_to_incident_mapping()
index_to_place_mapping = get_index_to_place_mapping()


class FinalModel(keras.Model):
    def __init__(self, trunk_model, incident_weights, place_weights):
        super(FinalModel, self).__init__()
        self.permute = Permute((2, 3, 1))
        self.cropped = keras.layers.experimental.preprocessing.RandomCrop(224, 224)
        self.permuteback = Permute((3, 1, 2))
        self.trunk_model = trunk_model
        self.incident_proj = Dense(len(incident_to_idx), input_shape=(1024,), name="incidents_projection", weights=incident_weights)
        self.places_proj = Dense(len(place_to_idx), input_shape=(1024,), name="places_projection", weights=place_weights)
       
    def from_pretrained(filepath):
        trunk = resnet.trunk()
        mdl = FinalModel(trunk, None, None)
        mdl.load_weights(filepath)
        return mdl

 
    def call(self, inputs, training=True):
        x = self.permute(inputs)
        x = self.cropped(x, training=training)
        x = self.permuteback(x)
        x = self.trunk_model(x, training=training)
        
        return self.incident_proj(x, training=training), self.places_proj(x, training=training)
