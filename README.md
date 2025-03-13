# Brain Tumor MRI Classification MLOps Project

This project implements an end-to-end MLOps pipeline for brain tumor MRI classification using TensorFlow and modern MLOps practices.

## Project Overview

The project uses a Convolutional Neural Network (CNN) to classify brain MRI scans into different tumor categories. It implements a complete MLOps pipeline including:

- Data versioning and preprocessing
- Model training and evaluation
- Model serving via REST API
- Continuous training pipeline
- Model monitoring and drift detection
- Automated testing and deployment

## Project Structure

```
├── src/
│   ├── data/          # Data processing and loading
│   ├── model/         # Model architecture and config
│   ├── training/      # Training scripts and utilities
│   ├── inference/     # Model serving and inference
│   ├── monitoring/    # Model monitoring and drift detection
│   └── pipeline/      # Training pipeline orchestration
├── tests/             # Unit and integration tests
├── configs/           # Configuration files
├── docker/            # Dockerfiles and compose files
└── docs/              # Documentation
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/parisa-nsh/Brain_Tumor_MRI_CNN.git
cd Brain_Tumor_MRI_CNN
```

2. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
```bash
python src/data/prepare_data.py --config configs/data_config.yaml
```

### Model Training
```bash
python src/training/train.py --config configs/train_config.yaml
```

### Model Serving
```bash
python src/inference/serve.py
```

## MLOps Features

- **Data Version Control**: Using DVC for data and model versioning
- **Experiment Tracking**: MLflow for experiment tracking and model registry
- **Model Monitoring**: Continuous monitoring with drift detection
- **CI/CD**: Automated testing and deployment pipeline
- **Containerization**: Docker containers for reproducible deployment
- **API Serving**: FastAPI for model serving
- **Configuration Management**: Hydra for managing configurations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Kaggle notebook by Jay Kumar
- Brain Tumor MRI dataset
- TensorFlow team and community
