# SVM-Face-Recognition
Utilizing a SVM to recognize faces

## PREP.PY
load the folder with the images and parse them into np arrays in which are preproccessed into training and testing sets
after they are split a model is made and trained with the data and dumped to a file to be used by the live camera file


## FACEDECT.PY 
This takes in live camera feed and parses into single frames in which are passed through the trained model to make a prediction 