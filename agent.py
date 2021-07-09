from actor import mainActor
from critic import mainCritic

import numpy as np
from numpy.random import choice
import random
from collections import namedtuple, deque


class mainReplayBuffer:
    def __init__(self, bufferSize, batchSize):
    
        self.memory = deque(maxlen=bufferSize)
        self.batchSize = batchSize
        self.experience = namedtuple("Experience", field_names=["state1", "action1", "reward1", "nextState1", "done1"])
    
    def plus(self, state1, action1, reward1, nextState1, done1):
        e = self.experience(state1, action1, reward1, nextState1, done1)
        self.memory.append(e)
    
    def general(self, batchSize=32):
        return random.sample(self.memory, k=self.batchSize)
    
    def __len__(self):
        return len(self.memory)
    
    
class mainAgent:
    def __init__(self, stateSize, batchSize, isEval = False):
        self.stateSize = stateSize
        self.actionSize = 3
        self.bufferSize = 1000000
        self.batchSize = batchSize
        self.memory = mainReplayBuffer(self.bufferSize, self.batchSize)
        self.inventory = []
        self.isEval = isEval
        
        self.gamma = 0.99
        self.tau = 0.001
        
        self.actorLocal = mainActor(self.stateSize, self.actionSize)
        self.actorTarget = mainActor(self.stateSize, self.actionSize)    

        self.criticLocal = mainCritic(self.stateSize, self.actionSize)
        self.criticTarget = mainCritic(self.stateSize, self.actionSize)
        
        self.criticTarget.model.set_weights(self.criticLocal.model.get_weights()) 
        self.actorTarget.model.set_weights(self.actorLocal.model.get_weights())
        
    def newAct(self, state1):
        maxOptions = self.actorLocal.model.predict(state1)
        self.lastState = state1
        if not self.isEval:
            return choice(range(3), p = maxOptions[0])
        return np.argmax(maxOptions[0])
    
    def newStep(self, action1, reward1, nextState1, done1):
        self.memory.plus(self.lastState, action1, reward1, nextState1, done1)
        if len(self.memory) > self.batchSize:
            newExperiences = self.memory.general(self.batchSize)
            self.newLearn(newExperiences)
            self.lastState = nextState1

    def newLearn(self, newExperiences):               
        newStates = np.vstack([e.state1 for e in newExperiences if e is not None]).astype(np.float32).reshape(-1,self.stateSize)    
        newActions = np.vstack([e.action1 for e in newExperiences if e is not None]).astype(np.float32).reshape(-1,self.actionSize)
        newRewards = np.array([e.reward1 for e in newExperiences if e is not None]).astype(np.float32).reshape(-1,1)
        newDones = np.array([e.done1 for e in newExperiences if e is not None]).astype(np.float32).reshape(-1,1)
        newNextStates = np.vstack([e.nextState1 for e in newExperiences if e is not None]).astype(np.float32).reshape(-1,self.stateSize)

        actionsNext = self.actorTarget.model.predict_on_batch(newNextStates)
        mainQtargetsNext = self.criticTarget.model.predict_on_batch([newNextStates, actionsNext])
        
        mainQtargets = newRewards + self.gamma * mainQtargetsNext * (1 - newDones)
        self.criticLocal.model.train_on_batch(x = [newStates, newActions], y=mainQtargets)
        
        actionGradients = np.reshape(self.criticLocal.get_action_gradients([newStates, newActions, 0]),(-1, self.actionSize))
        self.actorLocal.train_fn([newStates, actionGradients, 1])
        self.softUpdate(self.criticLocal.model, self.criticTarget.model)  
        self.softUpdate(self.actorLocal.model, self.actorTarget.model)

    def softUpdate(self, localModel, targetModel):
        localWeights = np.array(localModel.get_weights())
        targetWeights = np.array(targetModel.get_weights())

        assert len(localWeights) == len(targetWeights)

        newWeights = self.tau * localWeights + (1 - self.tau) * targetWeights
        targetModel.set_weights(newWeights)
