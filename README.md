# CSE 455 Sp23 Final: Keyboard Defender (Cats)

## Dependencies

Follow the [local install instructions for PyTorch](https://pytorch.org/get-started/locally/).
You will also need to install OpenCV, Pandas, and whatever else Python complains about.

## Dataset

Note: Training on this data actually worsens the performance of the program so this step is not necessary. I have only included it for completeness.

In the root directory of this project, create a folder called `data` and download the [annotated Oxford IIIT Pet Dataset from Kaggle](https://www.kaggle.com/datasets/julinmaloof/the-oxfordiiit-pet-dataset). Extract the `.zip` into a subdirectory of `data` called `Oxford_IIIT_Pet_Dataset`. The file structure should include these folders:

```
cse455-project
└── data
    └── Oxford_IIIT_Pet_Dataset
        ├── annotations
        |   └── trimaps
        |       |   Abyssinian_1.png
        |       |   ...
        └── images
            |    Abyssinian_1.jpg
            |    ...
```

After downloading the data, you can optionally modify the values in `train.py` and run `python train.py` to train the model. Monitoring data will be outputted to `runs/` after training, and the model will be saved as a `.pt` file.

## The KeyboardDefender Program

You can start the program by running `python main.py`. This will open a webcam stream with bounding boxes around keyboards and cats, as well as FPS info and the status of whether there is a cat on the keyboard. If a cat is detected near a keyboard, a popup window will display and continue to reappear if closed, until the cat has left.

To exit, focus the video window and press `q`. Alternatively, you can `ctrl + c` the program.