import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

try:
    from IPython.display import clear_output, display
except ImportError:
    clear_output = None
    display = None


def _refresh_figure(fig):
    """Refresh figures in both scripts and notebook backends."""
    plt.figure(fig.number)
    if clear_output is not None and display is not None:
        clear_output(wait=True)
        display(fig)
    else:
        plt.show(block=False)
    plt.draw()
    plt.pause(1e-3)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, keep_all=False):
        self.reset()
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.data is not None:
            self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TrainLossPlot(object):
    def __init__(self):
        self.loss_train = []
        self.fig = plt.figure()

    def update(self, loss_train):
        self.loss_train.append(loss_train)

    def plot(self):
        plt.figure(self.fig.number)
        plt.clf()
        plt.plot(np.array(self.loss_train))
        plt.title("Train loss / batch")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        _refresh_figure(self.fig)

class AccLossPlot(object):
    def __init__(self):
        self.loss_train = []
        self.loss_val = []
        self.acc_train = []
        self.acc_val = []
        self.fig = plt.figure()

    def update(self, loss_train, loss_val, acc_train, acc_val):
        self.loss_train.append(loss_train)
        self.loss_val.append(loss_val)
        self.acc_train.append(acc_train)
        self.acc_val.append(acc_val)
        plt.figure(self.fig.number)
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(np.array(self.acc_train), label="acc. train")
        plt.plot(np.array(self.acc_val), label="acc. val")
        plt.title("Accuracy / epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(np.array(self.loss_train), label="loss train")
        plt.plot(np.array(self.loss_val), label="loss val")
        plt.title("Loss / epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        _refresh_figure(self.fig)


def print_hyperparams(learning_rate, batch_size, optimizer_name, loss_name):
    print("Active hyperparameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Loss Function: {loss_name}")


def print_model_summary(model, input_size, device=None):
    try:
        from torchsummary import summary
    except ImportError as exc:
        raise ImportError("torchsummary is required for model summaries.") from exc

    device_name = str(device) if device else "cpu"
    if device_name not in {"cpu", "cuda"}:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    original_device = next(model.parameters()).device
    model.to(device_name)
    try:
        summary(model, input_size=input_size, device=device_name)
    finally:
        model.to(original_device)


def visualize_model_architecture(
    model, 
    input_size, 
    scale_xy=1.0, 
    scale_z=1.0, 
    background_color="white", 
    show_legend=True,
    font_path="/System/Library/Fonts/Helvetica.ttc",
    font_size=14
):
    try:
        import visualtorch
        from PIL import ImageFont
    except ImportError as exc:
        raise ImportError("visualtorch and Pillow are required for architecture visualization.") from exc

    # Load custom font for the legend to improve readability
    font = None
    if show_legend and font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            print(f"Font at {font_path} not found. Falling back to default PIL font.")

    original_device = next(model.parameters()).device
    model.to("cpu")
    image = None
    errors = []
    candidate_shapes = [input_size, (1,) + input_size]

    try:
        for shape in candidate_shapes:
            try:
                image = visualtorch.layered_view(
                    model,
                    input_shape=shape,
                    scale_xy=scale_xy,
                    scale_z=scale_z,
                    background_fill=background_color,
                    legend=show_legend,
                    font=font
                )
                break
            except Exception as exc:
                errors.append(exc)
                image = None
                
        if image is None:
            print("visualtorch could not render the architecture with the provided input size.")
            if errors:
                print(f"Last visualtorch error: {errors[-1]}")
            return None
    finally:
        model.to(original_device)

    if 'display' in globals():
        display(image)
        
    return image

def plot_learning_curves(history, best_epoch=None, figsize=(12, 4)):
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    if not train_loss or not val_loss:
        raise ValueError("history must include non-empty train_loss and val_loss.")

    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = best_epoch or history.get("best_epoch")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    if best_epoch:
        axes[0].axvline(best_epoch, color="red", linestyle="--", label="Best Epoch")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Acc")
    axes[1].plot(epochs, val_acc, label="Val Acc")
    if best_epoch:
        axes[1].axvline(best_epoch, color="red", linestyle="--", label="Best Epoch")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    return fig


def show_misclassified_grid(dataset, y_true, y_pred, class_names, grid=(2, 4), seed=None, mean=None, std=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mis_idx = np.where(y_true != y_pred)[0]

    if len(mis_idx) == 0:
        print("No misclassified samples to display.")
        return

    rows, cols = grid
    total = rows * cols
    rng = np.random.default_rng(seed)
    select = rng.choice(mis_idx, size=min(total, len(mis_idx)), replace=False)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, select):
        img, _ = dataset[idx]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            if img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))
            if mean is not None and std is not None and img.ndim == 3:
                mean_arr = np.array(mean).reshape(1, 1, -1)
                std_arr = np.array(std).reshape(1, 1, -1)
                img = img * std_arr + mean_arr
            img = np.clip(img, 0, 1)

        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        true_name = class_names[int(y_true[idx])]
        pred_name = class_names[int(y_pred[idx])]
        ax.set_title(f"True: {true_name} - Pred: {pred_name}")
        ax.axis("off")

    for ax in axes[len(select):]:
        ax.axis("off")

    plt.tight_layout()
    print("Misclassified Examples:")
    plt.show()