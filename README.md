# Monocular Depth Estimation using modified PSPNet
The whole project is written by Tensorflow, Python.
It based on the unsupervised methods.

The network is mainly based on the structure of PSPNet.
In order to reduce the total amount of parameters,
the feature extraction part uses modified ERFNet instead of ResNet-50.

These days I modified the former direct skip-connection, adding an attention-based mechanism, which can be easily understood by reading 'PSPNet_v3.py'. I haven't test its improvement compared to direct skip-connection, in future I will make more experiments on that.

The loss function about unsupervised learning is from:
https://github.com/mrharicot/monodepth

The model is trained on KITTI. The result shows that unsupervised learning method has better performance than supervised learning in real world's scene. Since the evaluating results on KITTI are not good enough compared to the results published on several papers, the checkpoint files are not provided now.

The model is mainly focus on outdoor driving scenes. Below is the current evaluation video based on current training result in the daylight (The upper half of the result is not stable, which will improved in future):
https://pan.baidu.com/s/1rWlTp6SPTRUQs3Tzng1O4A

* Usage for training

  1. Revise the data path in ReadData.py and lr_consistency_main.py

  2. Run "python lr_consistency_main.py" for random-initialized training, if pre-trained checkpoint is needed to recover former training process, use "--checkpoint_path=/your/checkpoint/path".
  
  3. You can change the batch size (--batch_size) and input size (--input_height, --input_width) for your training, but remembering to revise the input size in ReadData.py. (Sorry about my laziness to combine them since I use 512*256 resolution.)

* Usage for evaluation matrix
  
  1. The evaluation codes are in:
  https://github.com/mrharicot/monodepth/utils

  2. Some revisions are needed if Python 3 is used such as "class map". Check online for them.

* Usage for testing real-world performance
  1. Revise the checkpoint path in running_v2.py

  2. Run "python running_v2.py". The depth map has been colorized using "cv2.COLORMAP_JET" option.
