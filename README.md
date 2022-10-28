## Phone Finder

### About

Repository to train a phone finder, find normalied coordinates of a phone in a given image and evaluate the phone finderr

### Usage

For installation of the libraries and packaged, run the below command

```
$ pip install -r requirements.txt
```


For training the model, run the below command

```
$ python train_phone_finder.py
```

To evaluate the model trained on given set of image in a folder, run the below command

```
$ python evaluate_phone_finder.py ./folder_name
```

To predict normlalised coordinate on a given image, run the following command

```
$ python find_phone.py ./path_to_image
```


### Approach

I have used deep-learning based semantic segmentation to train phone image overlaid on background images. To train the deep neural network, I generated a synthetic image dataset that should eventually look closer to the images supplied in the dataset. I used this dataset to train a DeepLav3+ model to generate semantic masks for the detection of phone. Further, I used Jarvis Marching algorithm to generate a convex hull (and thus polygon points around the mask) and used the centroid of this polygon as indication of prediction of location of the phone.

A detailed report on the approach can be found by clicking [here](https://docs.google.com/document/d/1ziBdYydsb_SrgVoJ1SBRXgFfwIXS-Ni4Y2kbPInyRq0/edit?usp=sharing)


### Collab Notebooks

To generate the synthetic data and train the network, the code for the approach can be found [here](https://colab.research.google.com/drive/1mxhfYGMMuiix0SM-NHC4b9jkDDTRYxQO?usp=sharing)

To infer and evaluate the trained model, the code for the approach can be found by clicking [here](https://colab.research.google.com/drive/1FauiJcE5Rxz0Qa0DX19WoKBEJ9KqrxiG?usp=sharing)




