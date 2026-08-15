import os
import torch

from config import Config
from dataset import load_tn5000_split
from utils.preprocessing import preprocess_single_image
from network import ThyroidNet
from PIL import Image


def main():

    cfg = Config()
    device = torch.device(cfg.DEVICE)

    print("=" * 60)
    print("TN5000 CLASS-WISE DIRECT INFERENCE TEST")
    print("=" * 60)

    print("Device:", device)

    model_path = os.path.join(
        cfg.MODEL_DIR,
        "thyroidnet_best.pth"
    )

    bank_path = os.path.join(
        cfg.MODEL_DIR,
        "support_bank.pt"
    )

    model = ThyroidNet(cfg).to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    bank = torch.load(
        bank_path,
        map_location=device
    )

    support_feats = bank["features"].to(device)

    print(
        "Support bank:",
        support_feats.shape
    )

    test_files = load_tn5000_split(
        cfg.TN5000_DIR,
        "test"
    )

    print(
        "Total test files:",
        len(test_files)
    )

    benign = None
    malignant = None

    for image_path, label in test_files:

        if label == 0 and benign is None:
            benign = (image_path, label)

        if label == 1 and malignant is None:
            malignant = (image_path, label)

        if benign is not None and malignant is not None:
            break

    print("\nSelected test images:")

    print(
        "BENIGN:",
        benign
    )

    print(
        "MALIGNANT:",
        malignant
    )

    def test_image(image_path, true_label):

        print("\n" + "=" * 60)

        print(
            "Image:",
            image_path
        )

        print(
            "True label:",
            cfg.CLASS_NAMES[true_label]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        print(
            "Original size:",
            image.size
        )

        tensor = preprocess_single_image(
            image,
            cfg.IMAGE_SIZE
        ).to(device)

        print(
            "Tensor shape:",
            tensor.shape
        )

        with torch.no_grad():

            result = model.forward_inference(
                tensor,
                support_feats
            )

            probs = torch.softmax(
                result["logits"],
                dim=-1
            )

        predicted_label = probs.argmax(
            dim=-1
        ).item()

        print(
            "Logits:",
            result["logits"].cpu().numpy()
        )

        print(
            "Probabilities:",
            probs.cpu().numpy()
        )

        print(
            "Predicted:",
            cfg.CLASS_NAMES[predicted_label]
        )

        print(
            "U1:",
            float(result["u1"])
        )

        print(
            "U2:",
            float(result["u2"])
        )

        print(
            "Graph:",
            result["graph_type"]
        )

        print(
            "K:",
            result["k"]
        )

        if predicted_label == true_label:
            print("RESULT: CORRECT")
        else:
            print("RESULT: INCORRECT")

    test_image(
        benign[0],
        benign[1]
    )

    test_image(
        malignant[0],
        malignant[1]
    )

    print("\n")
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()