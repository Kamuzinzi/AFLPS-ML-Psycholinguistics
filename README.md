# Adaptive Focal Loss with Personality Stratification for Stably Mitigating Hard Class Imbalance in Multi-Dimensional Personality Recognition


## Dataset Details
- The datasets used in this project include:
    - [Essays Big5 Dataset](https://huggingface.co/datasets/jingjietan/essays-big5)
    - [Kaggle MBTI Dataset](https://huggingface.co/datasets/jingjietan/kaggle-mbti)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/adaptive-focal-loss.git
   cd adaptive-focal-loss
   ```
2. Create a virtual environment and install dependencies:
   ```bash
    conda create --name afl-env python=3.8
    conda activate afl-env
    pip install -r requirements.txt
   ```

3. **Download the datasets**  
    You can download the datasets manually or use the Hugging Face library.

    **Using Hugging Face Library**  
    -  **Install the `datasets` library:**
        ```bash
        pip install datasets
        ```
    - **Download the datasets:**
        ```python
        from datasets import load_dataset

        # Load MBTI Kaggle Dataset
        mbti_kaggle = load_dataset('jingjietan/kaggle-mbti')

        # Load Essays Big Five Dataset
        essays_big5 = load_dataset('jingjietan/essays-big5')

        # Save datasets to the `dataset/` directory
        mbti_kaggle.to_csv('dataset/mbti_kaggle.csv')
        essays_big5.to_csv('dataset/essays_big5.csv')
            ```


## Usage

### Getting started
1. Change directory to model training:
    ```bash
    cd model_training
    ```
2. Run the script:
    ```bash
    python train.py
    ```





