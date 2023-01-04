import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

data_path = "D:/shared/DTU/7. Semester/Machine Learning Operations/Own Exercises folder/data/corruptmnist/"

train_paths = [data_path + "train_" + str(i)+".npz" for i in range(5)]
test_path = data_path + "test.npz"

train_images = np.concatenate([np.load(f)["images"] for f in train_paths])
train_mean = np.mean(np.mean(train_images, axis = 0))
train_std = np.std(train_images)



class mnist(Dataset):
    def __init__(self, type, transform=None):

        if type == "train": 
            filepaths = train_paths
            images = [np.load(f)["images"] for f in filepaths]
            self.images = np.concatenate(images)
            labels = [np.load(f)["labels"] for f in filepaths]
            self.labels = np.concatenate(labels)
        else: 
            filepath = test_path
            self.images = np.load(filepath)["images"]
            self.labels = np.load(filepath)["labels"]
        self.transform = transform




    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):

        sample = {"image": self.images[index], "labels": self.labels[index]}

      #  print(self.images[index].shape)

        sample = self.transform(sample)

        return sample


class ToTensor_and_norm(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        image = torch.from_numpy(image)
        image = image.view(1,28,28)

        transformation = transforms.Compose([transforms.Normalize(mean=train_mean,
                             std=train_std)])
        transformed_image = transformation(image)
 #       print(torch.mean(torch.mean(transformed_image, dim = 1)))
        return {'image': transformed_image,
                'labels': labels
                }

#train_dataloader = DataLoader(mnist("train", transform = ToTensor_and_norm()), batch_size = 16, shuffle = False)

""" for sample in train_dataloader: 
    image = sample['image']
    labels = sample['labels']
    print(image.shape)
    print(labels.shape)
    break """
""" 
sample = next(iter(train_dataloader))
image = sample['image']

plt.imshow(image[0].permute(1, 2, 0).detach().numpy())
plt.show()
 """