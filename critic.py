from keras import layers, models, optimizers
from keras import backend as K


class mainCritic:

    def __init__(self, stateSize, actionSize):
        
        self.stateSize = stateSize
        self.actionSize = actionSize

        self.buildModel()

    def buildModel(self):

        # Define input layers
        newState = layers.Input(shape=(self.stateSize,), name='newState')
        newAction = layers.Input(shape=(self.actionSize,), name='newAction')

        netState = layers.Dense(units=16,kernel_regularizer=layers.regularizers.l2(1e-6))(newState)
        netState = layers.BatchNormalization()(netState)
        netState = layers.Activation("relu")(netState)

        netState = layers.Dense(units=32, kernel_regularizer=layers.regularizers.l2(1e-6))(netState)

        networkActions = layers.Dense(units=32,kernel_regularizer=layers.regularizers.l2(1e-6))(newAction)

        network = layers.Add()([netState, networkActions])
        network = layers.Activation('relu')(network)

        qValues = layers.Dense(units=1, name='q_values',kernel_initializer=layers.initializers.RandomUniform(minval=-0.003, maxval=0.003))(network)

        self.model = models.Model(inputs=[newState, newAction], outputs=qValues)

        adamOptimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=adamOptimizer, loss='mse')

        actionGradients = K.gradients(qValues, newAction)

        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=actionGradients)
