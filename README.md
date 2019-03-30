# Perceptual-Losses---Style-Transfer---Attention-Capsule-Network
Perceptual Losses - Style Transfer - Attention Capsule Network

This is a project for homework, we refer to attention53 and the capsule network to construct a new feedforward neural network to reproduce the style transfer method proposed in the 2016 ECCV paper "Perceptual Losses for Real-Time Style Transfer and Super-Resolution", and the new effect can highlight the significant features.

# References
1.[Perceptual Losses](https://arxiv.org/abs/1603.08155)
# Our Model
# Result
# How to use it？
Upload demo.ipynb to colab, and then run the ceils you need.
```
# train
s = Solver(trn_dir = '../Perceptual/pytorch_v/data',
           style_path = 'style/abs.jpg', 
           record_name = 'abstract_1_caps_record',
           result_dir = 'check', 
           weight_dir = './',
           num_epoch = 3,
           batch_size = 5,
           content_loss_pos = 1,
           lr = 1e-3,
           lambda_c = 1,
           lambda_s = 5e4, #5e4 1e6
           show_every = 20,
           save_every = 5000,
           pretrain = None,
           lossNet = 'vgg', # vgg senet50， 
           process_dir = 'process', 
           process_image = 'content/ybh.jpg', 
           process_scale = 0.3, 
           process_number  = 20, 
           record_number = 600,
           test_dir = '../Perceptual/pytorch_v/valid',
           test_number = 5,
           transNet = 'capsnet', # capsnet cnn
           opti = 'adamw', # adam adamw sgd
           norm_type = 'instance', # batch instance
           gram = 'gram' # gram gramP(Double Gram)
          )

s.train()

# test
content_name = 'tp.jpg'
test(
    weight_path='new_weight/udnie.weight' ,
    content_path='content/' + content_name, 
    output_path='fantasy_' + content_name.split('.')[0] + '.png',
    scale=0.9,
    transNet='capsnet',
    norm_type='instance', # batch instance
)
```
