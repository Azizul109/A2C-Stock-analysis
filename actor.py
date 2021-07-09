from keras import layers, models, optimizers
from keras import backend as K



class mainActor:
    
    
    def __init__(self, stateSize, actionSize):

        self.stateSize = stateSize
        self.actionSize = actionSize

        self.buildModel()

    def buildModel(self):
        
        newState = layers.Input(shape=(self.stateSize,), name='newState')
        
        network = layers.Dense(units=16,kernel_regularizer=layers.regularizers.l2(1e-6))(newState)
        network = layers.BatchNormalization()(network)
        network = layers.Activation("relu")(network)
        network = layers.Dense(units=32,kernel_regularizer=layers.regularizers.l2(1e-6))(network)
        network = layers.BatchNormalization()(network)
        network = layers.Activation("relu")(network)

        newActions = layers.Dense(units=self.actionSize, activation='softmax', name = 'newActions')(network)
        
        self.model = models.Model(inputs=newState, outputs=newActions)

        actionGradients = layers.Input(shape=(self.actionSize,))
        loss = K.mean(-actionGradients * newActions)

        adamOptimizer = optimizers.Adam(lr=.00001)
        updatesOp = adamOptimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, actionGradients, K.learning_phase()],
            outputs=[],
            updates=updatesOp)