
# Benchmarking `LLaVA-OneVision` with BASKET Dataset

This guide includes the code needed to benchmark the `LLaVA-OneVision` model with the BASKET dataset. Please follow the instructions for execution. 

### Setup and Execution Steps

1. **Download the BASKET Dataset**

2. **Clone the [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main) Repository**
    - Only single-modality part will be used for BASKET

3. **Convert the BASKET dataset into QA style**
    - Sample template: ""Analyze the video and rank the following 20 skills on a scale of 0 to 4. Follow the format skill:ranking. The skills are:..."

4. **Execute the `multi_gpu_inference.sh` script**
    - For training, add the correct last checkpoint path and modify the data input path to BASKET dataset

5. **Inferece with multi_gpu_inference.py script**
    - For inferece, directly run the provided script. 
    ```CUDA_VISIBLE_DEVICES=0,1,.. python test_skill.py```
