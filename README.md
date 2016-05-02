# MachineLearning-Spring-2016-VQA
 ***Abstract here***
 
Visual Question Answering
# Contibuters
* [@Jia Wang](https://github.com/waalwang)
* [@Yufeng Yuan](https://github.com/FrankeyYuan)
* [@Duo Liu](https://github.com/DuoL)

# Goal

Given a picture and a open-end question, output ***yes*** or ***no*** for this question.

Sample questions

<img src="/image/sample_image.png" width= "440" heigth="288">

question:Is the dog chasing the butterfly?

Answer: Yes

# Introduction
# Background
# Approach
<img src="/image/model.png" width="636", height="342">
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

For this run, we just define a default value for essential parameters, like training rate, iterations and batch size, and do the cross validation to get a rough picture of this project. The initial run wouldn’t show our job’s final performance, however, it provides some important data and clues for further work. Figure xxx shows the training loss and test loss of our initial run. The The smoothly decreasing training loss indicated that training rate’s value is suitable for this project. In addition, we get the initial accuracy 75.18% and turn point of test loss occurred at around #10000 iteration. 

Above data and clues are very important for our further work, such as parameter optimization and fine tuning. Firstly, we can follow the training rate’s value because in initial run it worked very well; Secondly, turn point of test loss indicates overfitting becomes terrible at that point, which provides us a sense how many iterations our further runs would have. Considering training runs and evaluations are very time-consuming, it can help us define efficient max iteration number, and then locate the turn point of test loss in further runs.

<img src="/image/initial_run_result.png" width="600", height="550">

## Parameter Optimization
Our initial run showed some very initial result, like 75.18% accuracy; verified some parameters workable, like 5e-4 learning rate making training loss smoothly decreasing. On the other hand, it showed we still had room to improve as well, like overfitting occurrence at iter #10000; And once overfitting negatively impacts training process, the training loss will no long true. So we need to optimize parameters to fix the negative effect of overfitting as possible; meanwhile, we need to try to optimize gradient descent to make training loss converge the optimal point continuously even at big training iterations.

Therefore, based on above analysis, we tried to do optimization in following 2 ways:
###Kill overfitting. We made it by using dropout.
Dropout means that we can temporarily remove some units out from the network along with all their incoming and outgoing connections, and the choice of which units to drop is random[1]. A ‘thinned’ network after applying dropout can prevent co-adaptation of units efficiently and then result in a more robust classifier. Figure xxx gives the comparison before and after applying dropout to a Neural Net. 

<img src="/image/dropout.png">

### Optimize gradient descent.
We chose RMSprop along with learning rate decay to implement the optimization. 

RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton, in fact it is identical to the first update vector of Adadelta that we derived above:

<img src="http://bit.ly/1UqYk02" align="center" border="0" alt="E[G^2]_t = 0.9E[g^2]_{t-1} + {0.1g^2}_t" width="254" height="22" />

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Ctheta_%7Bt-1%7D%20%3D%20%5Ctheta_t%20-%20%20%5Cfrac%7B%20%5Ceta%20%7D%7B%20%5Csqrt%7BE%5Bg%5E2%5D_t%2B%20%5Cvarepsilon%20%7D%20%7D%20g_t%0A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\theta_{t-1} = \theta_t -  \frac{ \eta }{ \sqrt{E[g^2]_t+ \varepsilon } } g_t" width="217" height="49" />

RMSprop as well divides the learning rate by an exponentially decaying average of squared gradients.

In our optimization process, we chose RMSprop algorithm with Learning Rate 5e-4 and Learning Rate Decay 0.99995, and default dropout value 0.5 for our #2 round training run, compared to initial run, the accuracy evaluated by cross validation was 79.59%, 4.7% improvement achieved.

##Fine-tuning

We then tried different dropout value(0.3, 0.2, and 0.1) to get better performance. Result showed that 0.2 dropout reached better accuracy compared to others, 80.02%. At this point, we tried to introduce another optimization algorithm “Adam” to get further. Adaptive Moment Estimation (Adam) is another method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients  like RMSprop, Adam also keeps an exponentially decaying average of past gradients , similar to momentum:

<img src="http://www.sciweavers.org/tex2img.php?eq=m_t%20%3D%20%5Cbeta_1m_%7Bt-1%7D%2B%281-%5Cbeta_1%29g_t&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="m_t = \beta_1m_{t-1}+(1-\beta_1)g_t" width="211" height="19" />

<img src="http://www.sciweavers.org/tex2img.php?eq=v_t%20%3D%20%5Cbeta_2v_%7Bt-1%7D%20%2B%20%281-%5Cbeta_2%29%7Bg%5E2%7D_t&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="v_t = \beta_2v_{t-1} + (1-\beta_2){g^2}_t" width="210" height="22" />

According to the properties of RMSprop and Adam, RMSprop has an excellent capability that deals with its radically diminishing learning rates; because of owning the property similar with Momentum additionally, Adam slightly outperform RMSprop towards the end of optimization as gradients become sparser. Therefore we chose replace RMSprop with Adam to do further optimization when the prior reaches the its highest performance. Figure xxx shows that this way achieved 80.06% accuracy that is the our best result so far.

<img src="/image/accuracy.png" width="471" height="512">



# Conclusion
# Acknowledgement
# Reference
> [1] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell,Dhruv Batra, C. Lawrence Zitnick, Devi Parik. VQA: Visual Question Answering, 2016

> [2] A Krizhevsky, I Sutskever, GE Hinton, ImageNet Classification with Deep Convolutional Neural Networks, Advances in Neural Information Processing Systems 25 (NIPS 2012)

> [3] K. Chatfield, K. Simonyan, A. Vedaldi, A. Zisserman, Return of the Devil in the Details: Delving Deep into Convolutional Nets. British Machine Vision Conference, 2014

> [4] Y Jia, E Shelhamer, J Donahue, S Karayev, Caffe: Convolutional architecture for fast feature embedding, Proceedings of the 22nd ACM international conference on Multimedia

> [5] C. Lawrence Zitnick. Bringing Semantics Into Focus Using Visual Abstraction, CVPR, 2013

> [6] N. Srivastava, G. Hinton, Dropout: A Simple Way to Prevent Neural Networks from Overftting. Journal of Machine Learning Research 15 (2014) 1929-1958



