# Attention Based Natural Language Grounding by Navigating Virtual Environment
Tensorflow code for ICML 2018 submission 

### Using the Environment
For running a random agent:
```
python env_test.py
```
To play in the environment:
```
python env_test.py --interactive 1
```
To change the difficulty of the environment (easy/medium/hard):
```
python env_test.py -d easy
```

### Training Gated-Attention A3C-LSTM agent
For training a A3C-LSTM agent with 32 threads:
```
python a3c_main.py --num-processes 32 --evaluate 0
```
The code will save the best model at `./saved/model_best`.

To test the trained model
```
python a3c_main.py --evaluate 2 --load saved/model_best --visualize 1
``` 
