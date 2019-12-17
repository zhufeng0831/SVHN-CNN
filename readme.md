Final Project - CS6476 - Classification and Detection using Convolutional Neural Networks:

**Google Drive Link to weights:** https://drive.google.com/open?id=1gXxFgreVXnuddnRfXK_DTqxQnaa3TeUm
 
    Custom Model : custom_model.json, weights.h5
    VGG16 scratch: vgg_16.json, vgg16.h5
    VGG16 Pre-trained: vgg_16_pretrained.json, vgg_16_pretrained.h5
    
    Please position the h5 and json files inside the src directory

__Correctly and incorrectly labelled examples in ./Data/test-images and ./Data/wrong-images respectively__

**YouTube Link to the demo video:** https://youtu.be/RHfe8vs8S0Q

**Youtube Link to the presentation:** https://youtu.be/0VwWikszq7E


**Dataset link (For training purposes only):** http://ufldl.stanford.edu/housenumbers/

    Format 2 data train_32X32.mat and test_32X32.mat goes into src/Data/Format2/
    Format 1 train and test data go into src/Data/Format2/train and src/Data/Format2/test respectively

**To Run the program:**
Inside the src directory, execute
    
    python run.py <path_to_file>

The output image will be saved as out_image.png in the current directory