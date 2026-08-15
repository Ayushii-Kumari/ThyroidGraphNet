import torch


class Config:

    DATA_DIR = "data"

    TN5000_DIR = "data/TN5000"

    DDTI_DIR = "data/DDTI"

    MODEL_DIR = "models"

    RESULTS_DIR = "results"


    CLASS_NAMES = [
        "benign",
        "malignant"
    ]

    NUM_CLASSES = 2


    IMAGE_SIZE = 256


    CONVNEXT_MODEL = "convnextv2_tiny.fcmae_ft_in22k_in1k"

    SWIN_MODEL = "swinv2_tiny_window8_256"

    FEATURE_DIM = 768


    MC_DROPOUT_SAMPLES = 20

    MC_DROPOUT_P = 0.3


    K_MIN = 3

    K_MAX = 10

    GAT_HEADS = 4

    GAT_HIDDEN = 256

    SUPPORT_BANK_SIZE = 400


    BATCH_SIZE = 16

    EPOCHS = 40

    LR = 1e-4

    WEIGHT_DECAY = 1e-5

    AUX_LOSS_WEIGHT = 0.3


    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )