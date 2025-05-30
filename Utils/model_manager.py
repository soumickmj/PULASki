#!/usr/bin/env python
'''

Purpose : model selector

'''
import sys
from ssl import SSLSocket
import torch.nn as nn
from Models.ProbUNetV2.model import InjectionConvEncoder2D, InjectionUNet2D, ProbabilisticSegmentationNet
from Models.attentionunet3d import AttU_Net
from Models.attentionunet2d import AttU_Net as AttU_Net2D
from Models.prob_unet.probabilistic_unet import ProbabilisticUnet
from Models.prob_unet2D.probabilistic_unet import ProbabilisticUnet as ProbabilisticUnet2D
from Models.unet3d import U_Net, U_Net_DeepSup
from Models.unet2d import U_Net as U_Net2D, U_Net_DeepSup as U_Net_DeepSup2D
from Models.SSN.models import StochasticDeepMedic
from Models.VIMH.unet_Ensemble import UNet_Ensemble as VIMH2D
from Models.VIMH.unet_Ensemble3D import UNet_Ensemble as VIMH3D
from Models.dounet2d import UNet as DOUNet2D
from Models.dounet3d import UNet as DOUNet3D
from Models.DPersona.DPersona import DPersona
from Models.CIMD.cimd_main import CIMD
from Models.BerDiff.berdiff_main import BerDiff
from Models.MrPrism.utils import get_network, Config

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2023, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

MODEL_UNET = 1
MODEL_UNET_DEEPSUP = 2
MODEL_ATTENTION_UNET = 3
MODEL_PROBABILISTIC_UNET = 4

def getModel(model_no, is2D=False, n_prob_test=0, prob_injection_at="end", no_outact_op=False, imsize=None): #Send model params from outside
    defaultModel = U_Net() #Default
    if is2D:
        if model_no not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            sys.exit(f"Invalid model ID {model_no} for 2D operations.")
        if model_no in [4]:
            print(f"Warning: Even though {model_no} has been implemented for 2D operations, it has bugs. Use with caution!")
        model_list = {
            1: U_Net2D(),
            2: U_Net_DeepSup2D(), 
            3: AttU_Net2D(),
            4: ProbabilisticUnet2D(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0),
            5: ProbabilisticSegmentationNet(in_channels=1, out_channels=1, 
                                            task_op=InjectionUNet2D,
                                            task_kwargs={"output_activation_op": nn.Identity if no_outact_op else nn.Sigmoid, "activation_kwargs": {"inplace": True}, "injection_at":  prob_injection_at},  
                                            prior_op=InjectionConvEncoder2D,
                                            prior_kwargs={"activation_kwargs": {"inplace": True}, "norm_depth": 2}, 
                                            posterior_op=InjectionConvEncoder2D,
                                            posterior_kwargs={"activation_kwargs": {"inplace": True}, "norm_depth": 2},
                                            ), 
            6: StochasticDeepMedic(input_channels=1, num_classes=2, scale_factors=((5, 5), (3, 3), (1, 1)), diagonal=True),
            # 7: VIMH(num_models=n_prob_test, mutliHead_layer="BDec2", num_in=1, num_classes=1), #This was the original plan. But the memory requirement is too high. So, we are using the following line instead (i.e. 4 models, like the original work).
            7: VIMH2D(num_models=4, mutliHead_layer="BDec2", num_in=1, num_classes=2),
            8: DOUNet2D(),
            9: DPersona(input_channels=1, num_classes=1, #Stage I
                        latent_dim=3, 
                        no_convs_fcomb=4,
                        num_experts=n_prob_test, 
                        reg_factor=0.00001,
                        original_backbone=False),
            10: DPersona(input_channels=1, num_classes=1, #Stage II
                        latent_dim=3, 
                        no_convs_fcomb=4,
                        num_experts=n_prob_test, 
                        reg_factor=0.00001,
                        original_backbone=False),
            11: CIMD(num_experts=n_prob_test, input_channels=1, num_classes=1),
            12: BerDiff(num_experts=n_prob_test, input_channels=1, num_classes=1),
            13: get_network(Config(imsize=imsize), Config(imsize=imsize).net, use_gpu=False, num_classes=1, n_raters=n_prob_test),
        }
    else:
        if model_no not in [1, 2, 3, 4, 5, 6, 7, 8]:
            sys.exit(f"Invalid model ID {model_no} for 3D operations.")
        if model_no in [4]:
            print(f"Warning: Even though {model_no} has been implemented for 3D operations, it has bugs. Use with caution!")
        model_list = {
            1: U_Net(),
            2: U_Net_DeepSup(), 
            3: AttU_Net(),
            4: ProbabilisticUnet(num_filters=[32,64,128,192]),
            # 4: ProbabilisticUnet(num_filters=[64,128,256,512,1024]),
            5: ProbabilisticSegmentationNet(in_channels=1, out_channels=1, 
                                            task_kwargs={"output_activation_op": nn.Identity if no_outact_op else nn.Sigmoid, "activation_kwargs": {"inplace": True}, "injection_at": prob_injection_at}, 
                                            prior_kwargs={"activation_kwargs": {"inplace": True}, "norm_depth": 2}, 
                                            posterior_kwargs={"activation_kwargs": {"inplace": True}, "norm_depth": 2},
                                            ) ,
            6: StochasticDeepMedic(input_channels=1, num_classes=2, scale_factors=((5, 5, 5), (3, 3, 3), (1, 1, 1)), diagonal=True),
            7: VIMH3D(num_models=4, mutliHead_layer="BDec2", num_in=1, num_classes=2),
            8: DOUNet3D()
        }
    model = model_list.get(model_no, defaultModel)
    
    if model_no == 5:
        model.init_weights(*[nn.init.kaiming_uniform_, 0])
        model.init_bias(*[nn.init.constant_, 0])
    
    # if model_no in [6,7,8]:
    #     print("Warning: The weights of the non-ProbUNet baselines are being initialised using kaiming. Be careful! It's not part of the original!")
    #     model.init_weights(*[nn.init.kaiming_uniform_, 0])
    #     model.init_bias(*[nn.init.constant_, 0])
        
    return model
