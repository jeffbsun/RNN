# RNN - a Recurrent Neural Network in Sports-betting
Betting markets tell us how well top analysts can predict events. These numbers can be used as benchmarks for the performance of predictive machine learning algorithms. 

[Image] optional
2 fighters matchup, with Vegas betting lines, and implied win probabilities

The for

## The challenge of MMA
[Image] optional
Chained, intertwined events

Mixed martial arts (MMA) was selected as the sport for RNN to predict. The sport has two-fighter fights, resulting in a Win, Loss, or Draw. The sport was chosen for its simple format and the following meaningful challenges:
* **Chained, intertwined events**: The strength of each fighter is determined by each one of his previous fights, and the strength of those opponents. 
* **One-on-one matchups**: The strength of each fighter is not a singular metric, but it also can represented as strengths or weakness against certain fighter types (like rock-paper-scissors). 
* **Modest training data**: The entire set of recorded mixed martial arts fights numbers only 290,000, with most missing auxiliary information. 
* **High variance**: For meaningful predictions, there needs to be a metric on the reliability, staleness, or variability of a metric, not just the metric itself. 

## Mathematical frameworks & benchmarks
The performance of RNN can be compared to existing mathematical frameworks and markets. 
**Elo**: The famous Chess-ranking system, which details a calculable number specifying player strength, and the win probability between two different Elo values. 
**Glicko**: An improvement on Elo, by Dr. Glickman, which adds a metric for variability (confidence) to each metric of strength. 
**Vegas line**: The aggregate prediction of top analysts. Each line consists of an _open_ and a _close_, with the _close_ being the more accurate. 
**RNN (recurrent neural network)**: 

## Tensorflow & training
Training was done with the [AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer). 
### Input data
* **Numerical traits**: fighter weight, height, age, time since last fight.
* **Categorical traits**: fighter home country, fight league, fight country.
* **Fighter strength vector**: each fighter’s strength vector are included in the input data.
### Ending input data
* Information about the result of a fight. This data is used to augment the fighter strength vectors after each fight, but not used to predict the outcome of the fight.
* **Traits**: outcome(win, loss), end round, method of victory, closing odds, significant strikes, fight statistics.
### Fighter strength vector
Each fighter’s strength was , with 8 being a hyper-parameter. 

[Image of structure of input & outputs]

### Other technical challenges
* One primary challenge was speeding up the training time on the intertwined events. Since Tensorflow performs batches quickly, it was important. This improved the training time by 20x. 
[Code sample or image]
* A custom web scraping engine was built to scrape historical MMA data from the websites of Sherdog, ESPN, Fightmetric, FightAnalytics, and BestFightOdds. (Chromium and HTTP requests)
[Code sample]
* Besides setting up the infrastructure. remaining time was spent on feature engineering.
[code?]



## Performance
The resulting error (loss) of RNN, as measured by mean-squared error, is comparable to opening and closing Vegas line. The training data comprised the first 80% of fights, and the following are results from the separate validation set. 
[Chart]


## RNN and other complex problems
Recurrent neural networks are used extensively in machine learning problems that involve chained repeated events. Interpreting text word-by-word is a prime example, where each word represents a step in the chain, influenced in meaning by the previous words, and influencing later words. Audio recognition and other time-series data also fall in the category where RNNs are useful. 




