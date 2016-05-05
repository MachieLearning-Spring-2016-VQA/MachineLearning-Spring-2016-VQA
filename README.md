# MachineLearning-Spring-2016-VQA
# Authors:

* [@Jia Wang](https://github.com/waalwang)       Email: jia_wang2@student.uml.edu
* [@Yufeng Yuan](https://github.com/FrankeyYuan) Email: Yufeng_Yuan@student.uml.edu
* [@Duo Liu](https://github.com/DuoL)            Email:Duo_Liu@student.uml.edu

<!--# Abstract-->
<!-- ***Abstract here***-->
 
<!--Visual Question Answering-->


# Introduction
The task of this project is given a abstract image and a natural language question about the image, output ***yes*** or ***no*** for this question. There will be 3000 test images and for each image, there will be three different questions about the image, and we are trying to get a higher accuarcy for test.
Figure 1 is a sample, the question for this image is "Is the dog chasing the butterfly?", test the image and question in our trained model, gives the answer "yes" for this question.

question:Is the dog chasing the butterfly?  Answer: Yes<br>
<img src="/image/sample_image.png" width= "440" heigth="288">

Figure 1. VQA Sample
<!--question:Is the dog chasing the butterfly?-->

<!--Answer: Yes-->

# Goal

Accuracy > 70%

# Background
Alex K. Et al.***[2]*** firstly accomplished the deep-level convolutional neuron network, after that Y.jia Et al.***[4]*** made it possible to Use Caffe deep learning framework to accelerate the optimization of the model, Stanislaw A. Et al.***[1]*** Combined image encoding and NLP to accomplish VQA including the result of extracting the semantics of image made by Lawrence et al.***[5]***
# Approach
* Our basic VQA model is according to VQA, the only modification depends on special case of this project---abstract images and ‘yes’ or ‘no’ answers. 
* The CNN we used for image encoding is a simplification of [VGG\_CNN\_F](https://gist.github.com/ksimonyan/a32c9063ec8e1118221a#file-readme-md)***[3]*** which has the same structure as Alexnet, but here, we made a little difference.
* The CNN model consists of 5-layer convolutional neuron network, 3 max pooling layers and 2 full-connected layers. 
	* The core of the first layer is 11×11, the stride is 4, and the total cores are 64. 
	* The second-layer core is 5×5, stride 1, the total are 256, and the last 3 layers cores are 3×3, total is 256, see figure2.5-Layer (CNN + SLTM).
* And there is a normalization layer called ReLu  that is in the back of each convolutional neuron network, which is used to normalize features.
* What’s more, the max pooling layer is located behind the 1st, 2nd  and the 5th layer, which are used to minimize the influence of transformation.
* The dimension of the 2-layer full-connected is 4096 and the output of the second-layer full-connected will be the final result of image encoding.
* After that,  we used 2-layer and 512 dimension RNN+LSTM network to build a NLP model based on VQA, the dimension of the final output is also 1024.
Finally both the image and question features are transformed into a common space and fused via the full-connected layer followed by a 2-layer softmax layer and we got the final ‘Yes’ or ‘No’ result of a question.


<img src="/image/model.png" width="636", height="342">

Figure 2.5-Layer (CNN + SLTM)<br>
## Specific training method:
We use cross-validation to evaluate the test loss because the ground truth test set is not available on kaggle competition. 
We can ignore the little difference on using cross-validation because we only randomly pick 500 images which are 3% of all the images.
During the cross-validation we use the recommended formula to calculate the accuracy, that is we use following formula to answer  a non-deterministic question.

![evaluation](/image/evaluation.png)

We found the key point to improvement after the first training and figured out a way to deal with this problem. We will talk about this in detail in the later chapters.

# Dataset
All Data are provided by [Kaggle Visual Question Answering](https://inclass.kaggle.com/c/visual-question-answering/data).
## Data Files

| Filename               | Format           | Description                     |
| -----------------------|----------------- | ------------------------------- |
| train_annotations      | .json(36.78 mb)  | all yes/no question answers     |
| train_images           | .zip (2.71 gb)   | 20,000 abstract training images |
| train_questions        | .zip (4.39 mb)   | 60,000 training questions       |
| test_images            | .zip (1.35 gb)   | 3,000 testing images            |
| test_questions         | .zip (2.13) mb   | 9,000 testing questions         |


# Evaluation
## Initial Run
After getting all data by preprocessing, we can start initial run. Since the CNN model we using is pre-trained by ILSVRC 2012, we need to define initial parameters according to following rules of thumb:

1.	We need to define a learning rate with small value, like 5e-4, otherwise, gradient descent may not be converged.

2.	we may check test loss by cross validation per 1000 iterations to find out the turn point.

3.	Considering the limited size of training set, we just randomly pick 500 images from training set as test set.

For this run, we just define a default value for essential parameters, like training rate, iterations and batch size, and do the cross validation to get a rough picture of this project. The initial run wouldn’t show our job’s final performance, however, it provides some important data and clues for further work. Figure 3 shows the training loss and test loss of our initial run. The The smoothly decreasing training loss indicated that training rate’s value is suitable for this project. In addition, we get the initial accuracy 75.18% and turn point of test loss occurred at around #10000 iteration. 

Above data and clues are very important for our further work, such as parameter optimization and fine tuning. Firstly, we can follow the training rate’s value because in initial run it worked very well; Secondly, turn point of test loss indicates overfitting becomes terrible at that point, which provides us a sense how many iterations our further runs would have. Considering training runs and evaluations are very time-consuming, it can help us define efficient max iteration number, and then locate the turn point of test loss in further runs.

<img src="/image/initial_run_result.png" width="600", height="550">
<br>Figure 3. Initial Run Result
## Parameter Optimization
Our initial run showed some very initial result, like 75.18% accuracy; verified some parameters workable, like 5e-4 learning rate making training loss smoothly decreasing. On the other hand, it showed we still had room to improve as well, like overfitting occurrence at iter #10000; And once overfitting negatively impacts training process, the training loss will no long true. So we need to optimize parameters to fix the negative effect of overfitting as possible; meanwhile, we need to try to optimize gradient descent to make training loss converge the optimal point continuously even at big training iterations.

Therefore, based on above analysis, we tried to do optimization in following 2 ways:
###Kill overfitting. We made it by using dropout***[6]***.
Dropout means that we can temporarily remove some units out from the network along with all their incoming and outgoing connections, and the choice of which units to drop is random[1]. A ‘thinned’ network after applying dropout can prevent co-adaptation of units efficiently and then result in a more robust classifier. Figure 4 gives the comparison before and after applying dropout to a Neural Net. 

<img src="/image/dropout.png">
<br>Figure 4. Dropout Result

### Optimize gradient descent.
We chose RMSprop along with learning rate decay to implement the optimization. 

RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton, in fact it is identical to the first update vector of Adadelta that we derived above:

![formula1](/image/formula1.png)

RMSprop as well divides the learning rate by an exponentially decaying average of squared gradients.

In our optimization process, we chose RMSprop algorithm with Learning Rate 5e-4 and Learning Rate Decay 0.99995, and default dropout value 0.5 for our #2 round training run, compared to initial run, the accuracy evaluated by cross validation was 79.59%, 4.7% improvement achieved.

##Fine-tuning

We then tried different dropout value(0.3, 0.2, and 0.1) to get better performance. Result showed that 0.2 dropout reached better accuracy compared to others, 80.02%. At this point, we tried to introduce another optimization algorithm “Adam” to get further. Adaptive Moment Estimation (Adam) is another method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients  like RMSprop, Adam also keeps an exponentially decaying average of past gradients , similar to momentum:

![formula2](/image/formula2.png)

According to the properties of RMSprop and Adam, RMSprop has an excellent capability that deals with its radically diminishing learning rates; because of owning the property similar with Momentum additionally, Adam slightly outperform RMSprop towards the end of optimization as gradients become sparser. Therefore we chose replace RMSprop with Adam to do further optimization when the prior reaches the its highest performance. Figure xxx shows that this way achieved 80.06% accuracy that is the our best result so far.

<img src="/image/accuracy.png" width="471" height="512">

##Result: 
We made overall 4% improvement through parameter optimization, and 1% improvement through fine-tuning. Since we did accuracy estimation by cross validation, and our calculation formula for accuracy is from official VQA website, the final result is slightly different as the ones shown in the above sections.


|Result/Methods|Parameters Optimization| Fine-tuning |Accuracy Official(Kaggle)|
|--------------|-----------------------|-------------|-------------------------|
|Original|X|X|75.18%(65.38%)|
|Mid-result|√|X|79.13%(69.40%)|
|Final|√|√|80.06%(70.99%)|





# Conclusion
<!--# Acknowledgement-->

1.	More data means better performance.

2.	Parameters optimization and fine-tuning leads to at least 5% improvement of  performance

3.	Real images + open ending = more interesting challenge.

# Team Roles

Jia Wang -------- model selection, parameter optimization/fine tuning proposal and implementation, coding for training, and paper work.

Duo Liu -------- project environment setup, experiments' data statistics and analysis, spreadsheet and paper work.

Yufeng Yuan -------- Dataset analysis, data preprocessing, coding for evaluation and other utilities, spreadsheet and paper work.

# Reference
> [1] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell,Dhruv Batra, C. Lawrence Zitnick, Devi Parik. VQA: Visual Question Answering, 2016

> [2] A Krizhevsky, I Sutskever, GE Hinton, ImageNet Classification with Deep Convolutional Neural Networks, Advances in Neural Information Processing Systems 25 (NIPS 2012)

> [3] K. Chatfield, K. Simonyan, A. Vedaldi, A. Zisserman, Return of the Devil in the Details: Delving Deep into Convolutional Nets. British Machine Vision Conference, 2014

> [4] Y Jia, E Shelhamer, J Donahue, S Karayev, Caffe: Convolutional architecture for fast feature embedding, Proceedings of the 22nd ACM international conference on Multimedia

> [5] C. Lawrence Zitnick. Bringing Semantics Into Focus Using Visual Abstraction, CVPR, 2013

> [6] N. Srivastava, G. Hinton, Dropout: A Simple Way to Prevent Neural Networks from Overftting. Journal of Machine Learning Research 15 (2014) 1929-1958
