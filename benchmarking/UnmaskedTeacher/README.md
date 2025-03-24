
# Benchmarking `Unmasked Teacher` with BASKET Dataset

This guide includes the code needed to benchmark the `Unmasked Teacher` model with the BASKET dataset. Please follow the instructions for execution. 

### Setup and Execution Steps

1. **Download the BASKET Dataset**

2. **Clone the [Unmasked Teacher](https://github.com/OpenGVLab/unmasked_teacher) Repository**

3. **Replace the scripts within video_sm with the provided dataset files.**
    - BASKET only accounts for top-1 accuracy. Remove any top-5 associated parts the from original repo

4. **Execute the `BASKET.sh` script**
    - For training, remove the ``--eval`` command and replace ``MODEL_PATH`` with the correct checkpoint for fine-tuning
    - For inference, directory run the .sh file with correct checkpoint and data path
```
bash BASKET.sh
```
