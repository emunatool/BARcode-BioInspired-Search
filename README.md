# BARcode: Biomimicry Search Engine

Welcome to BARcode (Biological Analogy Retriever), a search engine for biologically inspired design (BID). BARcode is introduced in the paper titled "Imitation of Life: A Search Engine for Biologically Inspired Design" presented at the AAAI 2024 conference. This project addresses the challenges of finding relevant biological solutions for real-world engineering challenges.

Paper: https://arxiv.org/abs/2312.12681
Authors: Hen Emuna, Nadav Borenstein, Xin Qian, Hyeonsu Kang, Joel Chan, Aniket Kittur, Dafna Shahaf
Conference: The 38th Annual AAAI Conference on Artificial Intelligence (AAAI 2024)

## About Biomimicry

Biomimicry, also known as biologically inspired design (BID), is a problem-solving methodology that applies analogies from nature to solve engineering challenges. By emulating nature's designs and processes, biomimicry seeks to inspire innovative solutions that are sustainable, efficient, and well-adapted to their environments.

## Important Folders

- **BARcode_search_engine**: Contains the code necessary to execute the search engine.
- **Baseline**: Includes the code and data required to generate baseline results, which are compared to the search engine's performance in the experiment (refer to section 4.1 in the paper).
- **Experiment_analysis**: Houses the results from the experiment detailed in section 4.1 of the paper. Additionally, it includes the code and data used to create the weighted score of the algorithm, as described in Appendix A.1.
- **Extracting_phrases**: This folder contains the code and data used to extract phrases for the matching process (see section 3.1 in the paper). Refer to point 4 in the "Getting Started" section below for further instructions.


## Getting Started

The code is implemented in python 3.10.5.

To get started with BARcode, follow these steps:

1. **Clone the Repository**: 
   ```
   git clone https://github.com/emunatool/BARcode-BioInspired-Search.git
   ```

2. **Set Up Virtual Environment (recommended)**: 
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Requirements**: 
   ```
   pip install -r requirements.txt
   ```

4. **Download Additional Files**: 
   - Download the necessary files from [Google Drive](https://drive.google.com/drive/folders/14lnFvnvY7VfgEc1obbSzo0PdlZYo0FIM?usp=sharing).
   - Unzip the files from the folder named "BARcode_search_engine_LFs".
   - Place the downloaded files into the corresponding folders within the cloned repository. For instance, move files from the unzipped "BARcode-BioInspired-Search" folder's "BARcode_search_engine" subfolder to the "BARcode_search_engine" folder in the repository.

**Note**: To use the "Extracting_phrases" module, copy the entire folder named "Extracting_phrases" into the main directory of the repository.

## Usage

### Running the Search Engine

To use the BARcode search engine:

1. Navigate to the `BARcode_speedup_version.py` file in `BARcode_search_engine` folder.
2. Locate the `run_query` function at the bottom of the script.
3. Example usage:
   ```python
   run_query(query_list=['collect water from air'], num_phrases_first_process=1000, num_phrases_sec_process=3000, top_n_results=15)
   ```
   - Provide the query or queries of interest in the `query_list` parameter.
   - Set the number of top results to display using the `top_n_results` parameter.
   - Choose whether to use the speedup version (`useSpeedUp=True`, default) or the version that runs on the entire dataset (`useSpeedUp=False`).

The output of BARcode is a CSV file containing four columns: "rank weighted score" (indicating the sentence's rank by the algorithm), "organism name" (the organism mentioned in the sentence), "sentence" (the potentially bioinspirational sentence), and "phrase" (the phrase used to match the sentence to the query). 

**Note**
Running BARcode for the first time may require initial setup time for configuring the embeddings.

**Possible Errors Due to Deprecated Models**:

When running the code, you may encounter errors related to deprecated models. One common error is:

### AttributeError: module 'numpy' has no attribute 'int'

This error may occur due to changes in the way certain data types are handled in newer versions of dependencies.

To resolve this error, follow these steps:

1. **Locate the Python File**: Navigate to the `modeling_deberta_v2.py` file within the `transformers\models\deberta_v2` directory in your Python environment.

2. **Edit the File**: Open the `modeling_deberta_v2.py` file in a text editor or IDE.

3. **Update the Code**: Look for instances of `np.int` in the file. This is likely where the error is originating. Change `np.int` to `int`.

4. **Save Changes**: Save the modifications to the `modeling_deberta_v2.py` file.

## Citation

If you use BARcode in your research or projects, we kindly request that you cite the following paper:

@inproceedings{emuna2024imitation,
  title={Imitation of Life: A Search Engine for Biologically Inspired Design},
  author={Emuna, Hen and Borenstein, Nadav and Qian, Xin and Kang, Hyeonsu and Chan, Joel and Kittur, Aniket and Shahaf, Dafna},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={503--511},
  year={2024}
}

