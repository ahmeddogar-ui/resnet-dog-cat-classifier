# ============================================
# VISUALIZATION FUNCTION
# ============================================
def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# ============================================
# GET A BATCH
# ============================================
print("Loading images...")
inputs, classes = next(iter(train_loader))

# ============================================
# MAKE A GRID
# ============================================
print("Creating grid...")
out = torchvision.utils.make_grid(inputs)

# ============================================
# DISPLAY
# ============================================
print("Displaying images...")
plt.figure(figsize=(12, 8))
imshow(out, title=[class_names[x] for x in classes])
plt.suptitle('Training Images with Data Augmentation', fontsize=16)
plt.show()

print("Close the window to continue...")
