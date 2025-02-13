## Final assignment for COMP309

Final assignment/Project was to build a CNN to do fruit classfication on three red fruit, Cherries, Tomatos and Strawberries.

Students had to do the full pipeline of processing, We erre provided with labeled data which was split into training and test, we were marked on the test portion of this split.

I choose to not use a pretrained model and built mine from nothing.

I started with pre-processing the labeled data and removing bad images. Following this I augmented the data with local production of synthetic images using an AI image generator.

Next the model was build, which was a four layer CNN with a varible learning rate and some extra things I am forgetting.

Testing against validation data was in the low 80s while tested against the actual test data resulted in a final accuracy of 76.90%
