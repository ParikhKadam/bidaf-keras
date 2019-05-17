# bidaf-keras
Implementation of [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603) in Keras 2

## What is this project about?
Machine Comprehension is a task in the field of NLP & NLU where the machine is provided with a passage and a question, and the machine tries to find an answer to the asked question from that given passage, by understanding the syntax and semantics of human language (here, English) and by establishing and understanding the relations betweeen the passage and the question.

We have implemented a model suggested in the paper Bidirectional Attention Flow for Machine Comprehension by a team of allennlp, popularly known as BiDAF.

Checkout this video to understand more:

[![Visualizing machine comprehension task with BiDAF](http://img.youtube.com/vi/ozifbOqihh8/0.jpg)](http://www.youtube.com/watch?v=ozifbOqihh8)

## What you can do with this project
- Train/Retrain this model with your own dataset.
- Use the pretrained model for extending your own model or to try this one out.
- Modify the code of this model to develop a new model architecture out of it.

## Prerequisites
- Python 3.6
- CUDA and cuDNN support for Tensorflow GPU (not mandatory, but it's better to have it)

## Installation
Execute this command `pip install bidaf-keras`

Note that the above code won't install tensorflow as there is no way to detecting if your system has GPU while installing this package. But you can explicitly mention if you want to install tensorflow (CPU/GPU) while installing this package.

- Installing Tensorflow CPU along with this package:
  
  Execute this command `pip install bidaf-keras[cpu]`

- Installing Tensorflow GPU along with this package:
  
  Execute this command `pip install bidaf-keras[gpu]`

## Usage
This project is available for use as a complete module. You can use this project via command-line arguments or by importing functionalities from it.:

- **Usage via command-line arguments:**
  
  To see the usage info, run `python3 -m bidaf --help` or `bidaf-keras --help`.

  Using this module via command-line arguments will provide you with limited functionalities. To get the real power, we recommend using it by importing its functionalities.
  
  The CLI mode will provide you with two basic functionalities: train and predict.
  
  - **Train:** To get usage information for training models, run `python3 -m bidaf train --help` or `bidaf-keras train --help`
    
    For example:
    
    ```
    python3 -m bidaf --model_name=bidaf_50.h5 --do_lowercase train --epochs=1 --steps_per_epoch=1 --validation_steps=1
    ```
    or
    ```
    bidaf-keras --model_name=bidaf_50.h5 --do_lowercase train --epochs=1 --steps_per_epoch=1 --validation_steps=1
    ```
  
  - **Predict:** To get usage information for running predictions on a model, run `python3 -m bidaf predict --help` or `bidaf-keras predict --help`
    
    For example:
     ```
     python3 -m bidaf --model_name=bidaf_50.h5 --do_lowercase predict --passage "This is tree." --question "What is this?" --return_char_loc --return_confidence_score
     ```
     or
     ```
     bidaf-keras --model_name=bidaf_50.h5 --do_lowercase predict --passage "This is tree." --question "What is this?" --return_char_loc --return_confidence_score
     ```
   
  **Note** that, some parameters like "--do_lowercase" are dependent on some other parameters like "--model_name". For example, if the model is trained to perform predictions on lowercase inputs, it might not work well if the inputs are not in lowercase.
  
  If you still have any queries on the need/use of any CLI argument, open an issue and we will reach you as soon as possible.
  
- **Usage by importing functionalities:**
  
  You can use the module by importing it in many different ways. So, we will just mention two example code snippets from which you will gain most of the knowledge required.
  
  - **Train:**
    ```
    from bidaf.models import BidirectionalAttentionFlow
    from bidaf.scripts import load_data_generators
    bidaf_model = BidirectionalAttentionFlow(400)
    bidaf_model.load_bidaf("/path/to/model.h5") # when you want to resume training
    train_generator, validation_generator = load_data_generators(24, 400)
    keras_model = bidaf_model.train_model(train_generator, validation_generator=validation_generator)
    ```
  - **Predict:**
    ```
    from bidaf.models import BidirectionalAttentionFlow
    bidaf_model = BidirectionalAttentionFlow(400)
    bidaf_model.load_bidaf("/path/to/model.h5")
    bidaf_model.predict_ans("This is a tree", "What is this?")
    ```
  **Note** that the above are the simplest code snippets. You can explore all functionalities available by looking at the parameters avaiable at different level. If you can't understand the parameters, just open an issue here. We may write the usage of all parameters in future if required.
    
  **Also note** that, in the above snippets, `bidaf_model` is just an object of class `BidirectionalAttentionFlow` and not the real Keras model. You can access the Keras model by using `keras_model = bidaf_model.model`.

## Features
- Supports both SQUAD-v1.1 and SQUAD-v2.0.
- Can predict answers from any length of passage and question but your memory should support it's size.
- Has multi-GPU support.
- Supports various embedding dimensions.
- Support for flexible answer span length.
- Support for fixed length passage and question.
- Ability to run predictions on a list of passages and questions.
- Sample shuffling can be enabled.
- Ability to return confidence score as well as character locations in the passage.
- Variable number of highway layers and decoders.

## Pre-trained Models
- **Model Name:** [bidaf_50.h5](https://drive.google.com/open?id=10C56f1DSkWbkBBhokJ9szXM44P9T-KfW)
  
  **Model Configuration:**
    - lowercase: True
    - batch size: 16
    - max passage length: None
    - max question length: None
    - embedding dimension: 400
    - squad version: 1.1

- **Model Name:** [bidaf_10.h5](https://drive.google.com/open?id=1CRxJ8IuPiXbbgQgtMVIztQpf0SIJcGR-)
  
  **Model Configuration:**
    - lowercase: True
    - batch size: 16
    - max passage length: None
    - max question length: None
    - embedding dimension: 400
    - squad version: 1.1

## Project flow
- First of all, the project will download the data required for training the model. This consists of two categories: Magnitude and SQuAD.
- SQuAD files are dataset files on which the model can be trained. Based on it's version specified, it will download the required files and preprocess them. The SQuAD data will be stored in **data/squad**. For more information on the SQuAD dataset, [visit this site](https://rajpurkar.github.io/SQuAD-explorer/).
- Magnitude files are a form of compressed and encoded files that we use for word embeddings at word-level and character-level. The project will download the required files based on the mebedding dimension specified by the user, in the **data/magnitude** directory. For more information on Magnitude, [visit this site](https://github.com/plasticityai/magnitude/)
- Now, users can train the model. The users have various options available while training to save models after every epoch and save the history of training as well. All these items will be saved in a directory called **saved_items**.
- To run predictions on any trained model, make sure that the model lies in the directory **saved_items** and then pass the name of the model to loading script. After loading the model, call the predict function.
- To resume training or to use pretrained models provided above, just place the model inside **saved_items** directory and pass it's name to the project.

## Improvements in future releases
- Support for Multiple (almost all) Languages
- Download pre-trained models automatically
- Support to specify model path instead of model name
- Provide two modes for preprocessing: Strict and Moderate.
- Measure accuracy of model with Moderate mode.
- GUI support
- Database support
- Your suggestions...?

## Warnings
- There's a dependency flaw in our project. The package we used to generate word embeddings, pymagnitude, hasn't yet added suport for fixed length inputs - https://github.com/plasticityai/magnitude/issues/50. Hence, trying to use this model with fixed length input will throw an error. We have looked into their source code and found the problem. We have forked that repo, fixed the issue and then generated a pull request but it seems like the author is no more maintaining that project. So, we have to wait. Till then, you may use the patched version of that package to make this functionality work. You can find it here - https://github.com/ParikhKadam/magnitude/tree/patch-1
- If the author is no more supporting this project, we are planning to maintain our forked version in future. But we don't know how to publish such a forked version on PyPI. You can help us in case you know it. Or wait until we learn xD.
- For now, you can install the patched version of this requirement by running this command:
  ```python3 -m pip install -U git+https://github.com/ParikhKadam/magnitude.git@patch-1```

## Issues
- Open:
  1. ...
- Solved:
  1. https://github.com/tensorflow/tensorflow/issues/24519
  2. https://github.com/keras-team/keras/issues/11978
  3. https://github.com/keras-team/keras/issues/12263

## Contributions
Thoughts, samples codes, modifications and any other type of contributions are appreciated. This is my first project in Deep Learning and also first being open source. I will need as much help as possible as I don't know the path I need to follow. Thank you..

## Donations
We want to continue maintaing this project with lots of new features. When we started this project, our end-to-end task was to provide exact answers to users' questions by searching contextual information automatically from the Internet. And we are surely moving towards achieving this task. But we are facing a big problem now. The GPU we used earlier was provided by our college and today is the day of our graduation. Happy to graduate but sad to know that we won't have a GPU to train this model on.. A big question is how will we continue? We haven't earned a single income to go buy a new one.. That's the reason we are taking donations.

Please help us if you can.. If you can't but you know someone who can, share this with them. We will be maintaining a list of people who donated and thus helped us. While donating, you can provide a link to your any profile (GitHub, Facebook, LinkedIn, etc..) and we will also add those links to the list we are maintaining here. Maybe it works for you as a kind of marketing.. We should also try helping as much as we can to the people who help us -- this is the reason we will be maintaining the list open on GitHub.

[![paypal](https://www.paypalobjects.com/en_GB/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=U83FVKU8MGK6J)

Thank you..

## My team
- [Dharanee Patel](https://github.com/dharaneepatel15/)
- [Zeel Patel](https://github.com/zeelp898/)

## Our Guide
- [Prof. Tushar A. Champaneria](https://github.com/tacldce/)

## Special Thanks to the researchers of BiDAF
- Sir, [Minjoon Seo](https://github.com/seominjoon/) for helping me out personally
- The team of allenai (http://www.allenai.org/)
