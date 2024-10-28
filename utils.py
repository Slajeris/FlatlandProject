import torch
import matplotlib.pyplot as plt

def train_model(model, train_loader, optimizer, criterion, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    print("Training complete!")


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device).float()
            labels = labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct_val / total_val

    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

def saliency(model, dataset, num_images, device):
    model.eval()

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(num_images):
        image, label = dataset[i]
        image_tensor = image.unsqueeze(0).to(device)
        image_tensor.requires_grad = True

        output = model(image_tensor)
        target_class = torch.argmax(output).item()

        model.zero_grad()
        output[0, target_class].backward()
        gradients = image_tensor.grad.data.abs()

        saliency = gradients.squeeze().cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

        axes[i].imshow(saliency, cmap='hot')
        axes[i].set_title(f'Saliency for Class {target_class}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()