import lightning.pytorch as pl
import torch

from config import config
from src import Light, TrafficSignsDataModule
from utils import logger, make_clear_directory, MyLogger
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from utils import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve

torch.set_float32_matmul_precision('medium')


def test():
    checkpoint_path = './output/checkpoints/best_checkpoint.ckpt'
    light = Light.load_from_checkpoint(
        checkpoint_path,
    )

    dm = TrafficSignsDataModule()

    light.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            images, labels = batch
            logits = light(images)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.append(labels)
            all_preds.append(preds)
            all_probs.append(probs[:, 1]) 

    all_labels = torch.cat(all_labels).cpu().numpy()
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_probs = torch.cat(all_probs).cpu().numpy()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    plot_confusion_matrix(all_labels, all_preds)
    plot_roc_curve(all_labels, all_probs)
    plot_precision_recall_curve(all_labels, all_probs)


def main():
    torch.cuda.empty_cache()
    MyLogger.init_loggers()

    test()


if __name__ == '__main__':
    main()
