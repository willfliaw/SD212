# Graph Learning (SD212) - 2023/2024

## Course Overview

This repository contains materials and resources for the course **SD212: Graph Learning**, part of the **Data & Artificial Intelligence** curriculum. The course covers the fundamental concepts and algorithms for analyzing and learning from graph data, including ranking, clustering, node classification, and link prediction. A significant portion of the course involves programming and applying these algorithms to real datasets.

### Key Topics:

- Graph Structures: Scale-free and small-world properties of real graphs.
- Ranking and Clustering: Techniques such as PageRank, clustering, and hierarchical clustering.
- Spectral Embedding: Representing graphs using eigenvalue decomposition.
- Graph Neural Networks (GNNs): Applying neural networks to graph data for classification and prediction tasks.

## Prerequisites

Students are expected to have:
- Basic knowledge of graph algorithms (e.g., search algorithms, shortest paths).
- Probability theory.
- Proficiency in Python programming.

## Course Structure

- Total Hours: 24 hours of in-person sessions (16 sessions), including:
  - 21 hours of lectures
  - 2 hours of quizzes
- Estimated Self-Study: 38.5 hours
- Credits: 2.5 ECTS
- Evaluation: Assessment through exams and quizzes. Attendance is mandatory for practical sessions.

## Instructor

- Professor Thomas Bonald

## Installation and Setup

Some exercises and projects require Python and relevant image processing libraries. You can follow the instructions below to set up your environment using `conda`:

1. Anaconda/Miniconda: Download and install Python with Anaconda or Miniconda from [Conda Official Site](https://docs.conda.io/en/latest/).
2. Image Processing Libraries: Create a new conda environment with the necessary packages:
   ```bash
   conda create -n graph python matplotlib numpy scipy scikit-image ipykernel pandas scikit-learn jupyter tqdm pytorch torchvision torchaudio pytorch-cuda=12.1 scikit-network dgl -c pytorch -c nvidia -c conda-forge -c dglteam/label/cu121
   ```
3. Activate the environment:
   ```bash
   conda activate graph
   ```

4. Launch Jupyter Notebook (if required for exercises):
   ```bash
   jupyter notebook
   ```

This environment will allow you to work with graph data, implement ranking and clustering algorithms, and use deep learning models like Graph Neural Networks (GNNs).

## How to Contribute

Feel free to contribute to the repository by:
- Submitting pull requests for corrections or improvements.
- Providing additional examples or extending the projects.
