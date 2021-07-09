from agent import mainAgent
from helper import stockData, getState, formatPrice

windowSize = 50
batchSize = 32
agent = mainAgent(windowSize, batchSize)
#mainData = stockData("^GSPC")
mainData = stockData("^IXIC")
#mainData = stockData("^GSPTSE")
#mainData = stockData("^RUT")
l_main = len(mainData) - 1
finalEpisodeCount = 60

for e in range(finalEpisodeCount):
    print("Episode " + str(e) + "/" + str(finalEpisodeCount))
    finalState = getState(mainData, 0, windowSize + 1)

    agent.inventory = []
    totalProfit = 0
    done1 = False
    for t in range(l_main):
        action = agent.newAct(finalState)
        actionProbability = agent.actorLocal.model.predict(finalState)

        nextState1 = getState(mainData, t + 1, windowSize + 1)
        reward1 = 0

        if action == 1:
            agent.inventory.append(mainData[t])
            print("buy:" + formatPrice(mainData[t]))

        elif action == 2 and len(agent.inventory) > 0:
            boughtPrice = agent.inventory.pop(0)
            reward1 = max(mainData[t] - boughtPrice, 0)
            totalProfit += mainData[t] - boughtPrice
            print("sell: " + formatPrice(mainData[t]) + "| profit: " + formatPrice(mainData[t] - boughtPrice))

        if t == l_main - 1:
            done1 = True
        agent.newStep(actionProbability, reward1, nextState1, done1)
        state = nextState1

        if done1:
            print("------------------------------------------")
            print("Total Profit: " + formatPrice(totalProfit))
            print("------------------------------------------")

#testData = stockData("^GSPC Test")
testData = stockData("^IXIC Test")
#testData = stockData("^GSPTSE Test")
#testData = stockData("^RUT Test")
l_test = len(testData) - 1
state = getState(testData, 0, windowSize + 1)
totalProfit = 0
agent.inventory = []
agent.isEval = False
done1 = False
for r in range(l_test):
    newAction = agent.act(state)

    nextState1 = getState(testData, t + 1, windowSize + 1)
    reward1 = 0

    if action == 1:

        agent.inventory.append(testData[t])
        print("buy: " + formatPrice(testData[t]))

    elif action == 2 and len(agent.inventory) > 0:
        boughtPrice = agent.inventory.pop(0)
        reward1 = max(testData[t] - boughtPrice, 0)
        totalProfit += testData[t] - boughtPrice
        print("sell: " + formatPrice(testData[t]) + " | profit: " + formatPrice(testData[t] - boughtPrice))

    if t == l_test - 1:
        done1 = True
    agent.step(actionProbability, reward1, nextState1, done1)
    state = nextState1

    if done1:
        print("------------------------------------------")
        print("Total Profit: " + formatPrice(totalProfit))
        print("------------------------------------------")
