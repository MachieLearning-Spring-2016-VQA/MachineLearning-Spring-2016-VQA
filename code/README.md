# Code copyrights:

1.	data/vqa_preprocessing.py 			---- author: VQA, modified by jia wang
2.	train_.lua 		 			---- author: VQA, modified by jia wang
3.	prepro_img.lua 					---- author: VQA, modified by jia wang
4.	prepro.py					---- author: VQA, modified by jia wang
5.	vqa_test_rate_fastest/vqa_test_rate_update.py	---- author: Yufeng Yuan
6.	misc/*						---- author: facebook torch team/VQA



# Installation:

0. before doing preprocessing job, you need to install torch7/python2.7/cuda and other dependencies at first.

	[torch7](http://torch.ch/)

	[cuda](https://developer.nvidia.com/cuda-zone)
	
	[python2.7](https://www.python.org/download/releases/2.7)

1. plz check [VQA official website](https://github.com/VT-vision-lab/VQA_LSTM_CNN) for preprocessing.
	
	
2. when all .h5 and .json files are ready, you can use below command to run training:

	th train.lua -backend nn -batch_size 100 -save_checkpoint_every 1000  -max_iters 40000
