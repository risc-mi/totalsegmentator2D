


def test_nnunet():
    from ts2d.core.util.util import is_nnunet_multilabel
    assert is_nnunet_multilabel(), ("The installation of nnunetv2 does not comprise the multilabel extension. "
                                    "Reinstall nnunetv2 from the local submodule at /libs/nnUNet-multilabel.")

#def test_pytorch():
#    import torch
#    assert torch.cuda.is_available(), "CUDA is not available. Please check your PyTorch installation."