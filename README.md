# ChestLens AI

## Description

Currently, radiologists manually analyze chest X-ray images to find any abnormalities, injuries, or diseases. This can be very time-consuming and like all things has the possibility of errors. To mitigate this issue we implemented an AI model to identify six common findings. These include Atelectasis, Cardiomegaly, Consolidation, Edema, No Finding, and Pleural Effusion. The main objective of using an AI model on chest X-rays is to help radiologists identify negative results or identify which of these diseases are present and the location of these diseases with the use of visual mapping.

## Getting started

1. **Clone this repository.**

2. **Install the required dependencies.**
    ```sh
    pip install -r requirements.txt
    ```

3. **Navigate to the client folder.**
    ```sh
    cd src/client
    ```

4. **Run the program.**
    ```sh
    python3 index.py
    ```

## References

1. **OpenCV**

    https://github.com/opencv/opencv-python

2. **PyTorch Grad Cam**

    https://github.com/jacobgil/pytorch-grad-cam

3. **TorchXRayVision**

    https://github.com/mlmed/torchxrayvision