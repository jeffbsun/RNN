# graphRNN - a Graph-Based Recurrent Neural Network in Sports Prediction
Here I describe a machine learning model that can predict the outcome of sporting events. The name of this project is called graphRNN, as it is a graph-shaped [recurrent neural network](https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf0912) for predicting the sport of MMA. Normally RNN cells are connected linearly (one input and one output), but in graphRNN, each cell (which represents one event) has two cells as its input and two cells as its output. 

## The challenge of MMA
<img width="1005" alt="Screen Shot 2019-05-25 at 5 18 00 PM" src="https://user-images.githubusercontent.com/3321825/58376276-b7712180-7f1b-11e9-80a4-84499e040007.png">
Our model graphRNN predicts mixed martial arts (MMA), a sport where each fight can be modeled as a two-player game, resulting in a win or loss. The sport was chosen for its simple format and the following meaningful challenges:

* *Graph of events*: The strength of each fighter is determined by each one of their previous fights, and the strength of each previous opponent. This has the effect of requiring the model to understand the full graph-shaped network of event history. 
* *One-on-one matchups*: The strength of each fighter is not a singular metric, but it also can represented as strengths or weakness against certain other fighter types. This contrasts with sports like horse racing, swimming, or track & field. 
* *Modest training data*: The entire set of recorded professional mixed martial arts fights numbers 294,000, with most missing auxiliary information. 
* *High variance*: For meaningful predictions, there needs to be a metric on the reliability, staleness, or variability of a metric, not just the metric itself. This requires the model to generalize well. 

## Mathematical frameworks & benchmarks
The performance of graphRNN can be compared to existing mathematical frameworks and benchmarks:

* *Elo*: This Chess-ranking system details a calculable rating ([Elo](https://en.wikipedia.org/wiki/Elo_rating_system)) specifying player strength, and the win probability between two different Elo values. 
* *Glicko*: An improvement on Elo, called [Glicko](http://www.glicko.net/glicko/glicko.pdf), which adds a metric for variability (confidence) to each metric of strength. 
* *Vegas line*: The [aggregate prediction](https://www.bestfightodds.com) of top analysts. Each line consists of an *open* and a *close*, with the *close* being the more accurate. 
* *graphRNN (graph recurrent neural network)*: Our model has the potential of storing more meaningful information about player strength, while traversing the graph of fight history. 

## Layers, structure & training
<img width="1599" alt="Screen Shot 2019-05-25 at 4 43 01 PM" src="https://user-images.githubusercontent.com/3321825/58376277-ba6c1200-7f1b-11e9-96d8-60ec78323108.png">
The above diagram details the structure of one cell of the graphRNN. One cell represents one fight between two fighters. The graphRNN cell are connected to each other through the fighter strength vector (one unit’s output strength vector is another unit’s input strength vector). In contrast with normal recurrent neural networks, each graphRNN cell has two cells as its input and two cells as its output. The two input cells are the two previous fights by each fighter, and the two output cells are the two subsequent fights by each fighter. Chaining all cells results in a graph of the entire fight MMA history.  

### Fighter strength vector
Each fighter’s strength is described as a vector. The vector is initialized as zeroes, and during training, the model decides what values to place at which locations in the vector, based on predictive ability. The strength vectors are then updated after each fight, with the winning and losing fighter each receiving certain updates. 
### Fight input data
Information about the fight that we know before the fight begins - this includes:
* *Numerical traits*: fighter weight, height, age, time since last fight.
* *Categorical traits*: fighter home country, fight league, fight country.
* *Fighter strength vector*: each fighter’s strength vector are included in the input data.
### Fight result data
Information about the result of a fight. This data is used to augment the fighter strength vectors after each fight, but not used to predict the outcome of the fight.
* *Traits*: outcome(win, loss), end round, method of victory, closing odds, significant strikes, fight statistics.
### Symmetry
The model does have a built-in concept of symmetry, where the layer nodes are mirrored between fighters A and B, and each fight is trained along with its mirror image. Other than symmetry, the model does not have knowledge of the space - all layer nodes are initialized randomly. 
### Training
Training was performed using `tensorflow 1.11.0` on an iMac Pro. The optimizer’s loss function was mean-squared error between the prediction and actual result. The model trained on the data set repeatedly, with the  learning rate decreasing over time. 

## Performance
The resulting error (loss) of graphRNN, as measured by mean-squared error, is 0.2114, which is comparable to the Vegas line. This shows that a well-structured machine learning model can perform at comparable levels to top analysts, and greatly outperforms existing mathematical frameworks. A loss of 0.2114 means that graphRNN predicts even-odds fights with an accuracy of 54%, which is better than the 53.4% accuracy of the opening Vegas line. 
<img width="1397" alt="Screen Shot 2019-05-25 at 9 09 32 PM" src="https://user-images.githubusercontent.com/3321825/58377669-bb606c00-7f3a-11e9-9143-bbc2e1dadbe8.png">
The training data comprised the first 80% of the fight history, and the above are results from the separate validation set. 

## Concepts
* [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)
* [Dense layer](https://www.tensorflow.org/api_docs/python/tf/layers/Dense)
* [Sigmoid](https://www.tensorflow.org/api_docs/python/tf/math/sigmoid)
* [Relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu)
* [Dropout](https://www.tensorflow.org/api_docs/python/tf/layers/Dropout)
* [Loss (mean squared error)](https://www.tensorflow.org/api_docs/python/tf/losses/mean_squared_error)
* [Feature column](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column)
* [RNN cell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/BasicRNNCell)
* [LSTM cell](https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/BasicLSTMCell)

### Other technical challenges
* One primary challenge was speeding up the training time on the intertwined events. Since Tensorflow performs batches quickly, it was important to batch together the training of fights that were independent of each other. Therefore each batch corresponded to a level in the fight history tree. Doing this improved the training time by 20x. 
```
# Calculate the level in the fight tree for this fight. 0 is bottom level (leafs of tree)
fightLevel = max(f0Level, f1Level)
level = levels[fightLevel]
level.append((fight, f0, f1))
```
* Since the graph/tree structure of RNN units was not standard to most machine learning problems, a special save & restore path was need to restore node weights from one run to the next. 
```
for i, savedValue in enumerate(saved):
    savedVar = tf.convert_to_tensor(savedValue)
    assignments.append(tf.assign(trainables[i], savedVar))
sess.run(assignments)
```
* A custom web scraping engine was built to scrape historical MMA data from the websites of Sherdog, ESPN, Fightmetric, FightAnalytics, and BestFightOdds. (Selenium and HTTP requests)
```
header = {‘User-Agent’: ‘Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11’}
req = urllib.request.Request(url, headers=header)
usock = urllib.request.urlopen(req)
```
* Many experiments were run to tune hyperparameters and feature engineering. 
