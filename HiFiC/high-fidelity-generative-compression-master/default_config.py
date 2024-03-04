#!/usr/bin/env python3

"""
Default arguments from [1]. Entries can be manually overriden via
command line arguments in `train.py`.

[1]: arXiv 2006.09965
"""
#TODO verifier si c'est bien nécéssaire, pour le moment, cette classe est dans perceptual loss et uniquement la
"""class imageType(object):
    UINT8 = {"np_type":np.uint8,"cent":1.,"range":255.,"factor":255./2.}
    UINT16 = {"np_type":np.uint16,"cent":1.,"range":65535.,"factor":65535./2.}
    #UINT32 = [np.uint32,,4294967295]
    #UINT64 = [np.uint64,,18446744073709551615]"""
class ModelTypes(object):
    COMPRESSION = 'compression'
    COMPRESSION_GAN = 'compression_gan'

class ModelModes(object):
    TRAINING = 'training'
    VALIDATION = 'validation'
    EVALUATION = 'evaluation'  # actual entropy coding

class Datasets(object):
    # path are made from high-fidelity-generative-compression
    # (you have to cd high-fidelity-generative-compression> before using train.py

    OID7_RGB8_10 = 'OID7_RGB8_10'
    OID7_RGB8_100 = 'OID7_RGB8_100'
    OID7_RGB8_1000 = 'OID7_RGB8_1000'
    OID7_RGB8_10000 = 'OID7_RGB8_10000'

    OID7_L8_100 = 'OID7_L8_100'
    OID7_L8_1000 = 'OID7_L8_1000'
    OID7_L8_10000 = 'OID7_L8_10000'

    OPENIMAGES = 'openimages'
    CITYSCAPES = 'cityscapes'
    JETS = 'jetimages'

class DatasetPaths(object):

    OID7_RGB8_10 = 'data/datasets/OID7_RGB8_10'
    OID7_RGB8_100 = 'data/datasets/OID7_RGB8_100'
    OID7_RGB8_1000 = 'data/datasets/OID7_RGB8_1000'
    OID7_RGB8_10000 = 'data/datasets/OID7_RGB8_10000'

    OID7_L8_100 = 'data/datasets/OID7_L8_100'
    OID7_L8_1000 = 'data/datasets/OID7_L8_1000'
    OID7_L8_10000 = 'data/datasets/OID7_L8_10000'

    OPENIMAGES = 'data/openimages'
    CITYSCAPES = ''
    JETS = ''

class directories(object):
    experiments = 'experiments'

class args(object):
    """
    Shared config
    """
    name = 'OID7_L8_100'
    silent = True
    n_epochs = 8 #each epoch will train images on (number of images in train/4, ex : 2500/epoch for 10000 img training dataset
    n_steps = 1e6
    batch_size = 4
    log_interval = 1000#images between each log, need to have enough images in validation to work (otherwise create a next() iter error.
    save_interval = 50000
    gpu = 0
    multigpu = True
    dataset = Datasets.OID7_L8_100
    dataset_path = DatasetPaths.OID7_L8_100
    shuffle = True

    # GAN params
    discriminator_steps = 0
    model_mode = ModelModes.TRAINING
    sample_noise = False
    noise_dim = 32

    # Architecture params - defaults correspond to Table 3a) of [1]
    latent_channels = 220
    n_residual_blocks = 9           # Authors use 9 blocks, performance saturates at 5
    lambda_B = 2**(-4)              # Loose rate
    k_M = 0.075 * 2**(-5)           # Distortion
    k_P = 1.                        # Perceptual loss
    beta = 0.15                     # Generator loss
    use_channel_norm = True
    likelihood_type = 'gaussian'    # Latent likelihood model
    normalize_input_image = False   # Normalize inputs to range [-1,1]
    
    # Shapes
    crop_size = 256
    image_dims = (1,crop_size,crop_size)
    latent_dims = (latent_channels,16,16)
    
    # Optimizer params
    learning_rate = 1e-4
    weight_decay = 1e-6

    # Scheduling
    lambda_schedule = dict(vals=[2., 1.], steps=[50000])
    lr_schedule = dict(vals=[1., 0.1], steps=[500000])
    target_schedule = dict(vals=[0.20/0.14, 1.], steps=[50000])  # Rate allowance
    ignore_schedule = False

    # match target rate to lambda_A coefficient
    regime = 'low'  # -> 0.14
    target_rate_map = dict(low=0.14, med=0.3, high=0.45)
    lambda_A_map = dict(low=2**1, med=2**0, high=2**(-1))
    target_rate = target_rate_map[regime]
    lambda_A = lambda_A_map[regime]

    # DLMM
    use_latent_mixture_model = False
    mixture_components = 4
    latent_channels_DLMM = 64

"""
Specialized configs
"""

class mse_lpips_args(args):
    """
    Config for model trained with distortion and 
    perceptual loss only.
    """
    model_type = ModelTypes.COMPRESSION

class hific_args(args):
    """
    Config for model trained with full generative
    loss terms.
    """
    model_type = ModelTypes.COMPRESSION_GAN
    gan_loss_type = 'non_saturating'  # ('non_saturating', 'least_squares')
    discriminator_steps = 1
    sample_noise = False
