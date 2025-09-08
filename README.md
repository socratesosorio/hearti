# HeartAI - 3D Cardiac MRI Analysis Platform (TreeHacks 2025 Most Impactful Grand Prize)

A comprehensive machine learning platform for 3D cardiac MRI analysis, featuring deep learning-based heart segmentation, embedding generation, and intelligent diagnosis assistance through vector search and LLM integration.

## 🎯 Overview

HeartAI combines advanced 3D computer vision, vector embeddings, and knowledge retrieval to provide automated cardiac MRI analysis. The platform can process NIfTI cardiac imaging data, generate intelligent embeddings, perform similarity search across medical records, and provide AI-assisted diagnosis with medical literature references.

## 🏗️ Architecture

### Core ML Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   3D UNet3D     │───▶│  Embedding Gen   │───▶│  Vespa Search   │
│  Segmentation   │    │  (CNN+Transformer)│    │   Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Heart Region    │    │    512-dim       │    │ Similar Patient │
│ Segmentation    │    │   Embeddings     │    │    Records      │
│ (8 chambers)    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ 3D Visualization│    │   FastAPI Endpoints  │    │ AI Diagnosis    │
│ (Plotly/Three.js)│    │   /upload, /search   │    │ (Perplexity)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🧠 Machine Learning Pipeline

### 1. 3D Heart Segmentation (UNet3D)

**Location**: `ml/embeder.ipynb`

The core segmentation model uses a 3D U-Net architecture designed specifically for cardiac MRI volumes:

- **Input**: NIfTI files (3D cardiac MRI volumes)
- **Output**: 8-class segmentation mask for different heart regions
- **Architecture**: 
  - Encoder: 4 levels with DoubleConv3D blocks (64→128→256→512 channels)
  - Bottleneck: 1024 channels with 3D dropout regularization
  - Decoder: Skip connections with transpose convolutions
  - Final: 1x1x1 conv for 8-class output

**Heart Regions Segmented**:
1. Left Ventricle
2. Right Ventricle  
3. Left Atrium
4. Right Atrium
5. Aorta
6. Pulmonary Artery
7. Superior Vena Cava
8. Inferior Vena Cava

**Training Details**:
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.0001)
- Data: 60 patients, 50 train / 10 test split
- Preprocessing: Cubic padding + resize to 128³ voxels
- Regularization: Dropout3D in bottleneck

### 2. Embedding Generation

**Location**: `api2/embeddings.py`

A sophisticated 3D CNN + Transformer architecture that converts cardiac MRI volumes into meaningful 512-dimensional embeddings:

```python
Architecture:
├── 3D CNN Feature Extraction
│   ├── Conv3d(1→64) + ReLU + MaxPool3d(2)
│   ├── Conv3d(64→128) + ReLU + MaxPool3d(2)  
│   ├── Conv3d(128→256) + ReLU
│   └── Conv3d(256→512) + ReLU
├── Spatial Flattening & Linear Projection
│   └── Linear(16384 → 512)
├── Transformer Encoder
│   ├── 6 layers of TransformerEncoderLayer
│   ├── 8 attention heads, 2048 FFN dimension
│   └── Batch-first processing
└── Global Pooling & Output
    ├── AdaptiveAvgPool1d(1)
    └── 512-dim embedding output
```

**Key Features**:
- **Multi-scale feature extraction**: Hierarchical CNN captures both local cardiac structures and global anatomy
- **Attention mechanisms**: Transformer layers learn spatial relationships between heart regions
- **Robust preprocessing**: MONAI transforms handle NIfTI loading, intensity scaling, and standardization
- **Base64 support**: Direct processing from web uploads without file system storage

### 3. Vector Search & Retrieval

**Location**: `api2/vespa-app/schemas/clinical_data.sd`

Uses Vespa for high-performance similarity search across cardiac embeddings:

```vespa
schema clinical_data {
  document clinical_data {
    field pat type string               # Patient ID
    field data type string             # Clinical data CSV string  
    field image_embedding type tensor<float>(d[512]) {
      attribute {
        distance-metric: angular      # Cosine similarity
      }
    }
  }
  
  rank-profile default {
    inputs {
      query(query_vec) tensor<float>(d[512])
    }
    first-phase {
      expression: closeness(image_embedding)  # ANN search
    }
  }
}
```

**Search Process**:
1. Input MRI → Embedding generation (512-dim)
2. ANN search in Vespa using angular distance (cosine similarity)
3. Retrieve top-k most similar patient records
4. Return clinical data for AI diagnosis

### 4. 3D Visualization Pipeline

**Location**: `modeling/vis.py`, `modeling/visualize.ipynb`

Advanced 3D rendering system for heart segmentation visualization:

**Components**:
- **Voxel Processing**: Converts segmentation masks to 3D coordinate arrays
- **Surface Extraction**: Marching Cubes algorithm for smooth mesh generation
- **Gaussian Smoothing**: σ=1.0 for artifact reduction and natural surfaces
- **Interactive Visualization**: Plotly/Three.js for web-based 3D exploration

**Visualization Types**:
1. **Raw Voxel Scatter**: Direct point cloud visualization with intensity mapping
2. **Surface Mesh**: Smooth reconstructed surfaces with material properties
3. **Multi-region Rendering**: Color-coded anatomical structures with toggle controls
4. **Cross-sectional Views**: Slice-based exploration with intensity mapping

## 🚀 Installation & Setup

### Prerequisites

```bash
# Python 3.8+
# CUDA-capable GPU (recommended)
# Docker (for Vespa)
```

### Core Dependencies

```bash
# ML & Computer Vision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai nibabel scikit-image scipy

# Embeddings & NLP  
pip install sentence-transformers transformers

# Vector Database
pip install pyvespa

# API & Web
pip install fastapi uvicorn python-dotenv

# Visualization
pip install plotly matplotlib pandas
```

### Environment Setup

1. **Clone and Setup**:
```bash
git clone <repository-url>
cd hearti
```

2. **Install Python Dependencies**:
```bash
# API 1 (Legacy)
cd api && pip install -r requirements.txt

# API 2 (Main)  
cd ../api2 && pip install -r requirements.txt
```

3. **Setup Vespa Database**:
```bash
cd api2/vespa-app
docker run -d --name vespa --hostname vespa \
  -p 8080:8080 vespaengine/vespa:latest
  
# Wait for startup, then deploy schema
./start-vespa.sh
```

4. **Environment Variables**:
```bash
# Create .env file
echo "PERPLEXITY_API_KEY=your_api_key_here" > .env
```

5. **Data Preparation**:
```bash
# Place your NIfTI cardiac data in:
mkdir -p data/25226366/cropped/
# Add your pat{N}_cropped.nii.gz files

# Place clinical CSV:
cp your_clinical_data.csv data/hvsmr_clinical.csv
```

## 💻 Usage

### 1. Start the ML API Server

```bash
cd api2
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Ingest Training Data

```bash
# This will process NIfTI files and generate embeddings
curl -X POST "http://localhost:8000/ingest" 
```

### 3. Process New Cardiac MRI

**Via API**:
```python
import base64
import requests

# Load your NIfTI file
with open("patient_scan.nii", "rb") as f:
    nii_base64 = base64.b64encode(f.read()).decode()

# Upload for analysis
response = requests.post("http://localhost:8000/upload", 
    json={"nii_path": nii_base64})

result = response.json()
print(f"Diagnosis: {result['diagnosis_text']}")
print(f"Confidence: {result['confidence']}")
```

**Via Jupyter Notebook**:
```python
# See ml/embeder.ipynb for training
# See modeling/visualize.ipynb for visualization examples

from embeddings import NIfTIToEmbedding

# Generate embedding
embedder = NIfTIToEmbedding()
embedding = embedder("path/to/scan.nii")
print(f"Generated embedding shape: {embedding.shape}")
```

### 4. 3D Visualization

```python
from modeling.vis import plot_smooth_heart, tensor_to_3d_points
import nibabel as nib
import numpy as np

# Load segmentation data
data_seg = np.array(nib.load("pat_segmentation.nii.gz").get_fdata())
data_cropped = np.array(nib.load("pat_cropped.nii.gz").get_fdata())

# Generate 3D visualization
processed_data, processed_colors = tensor_to_3d_points(data_seg, data_cropped)
plot_smooth_heart(processed_data, export_html="heart_3d.html")
```

## 🔌 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload NIfTI file for analysis |
| `/search` | POST | Text-based similarity search |
| `/healthcheck` | GET | Service status |

### Upload Endpoint Details

**POST `/upload`**

```json
{
  "nii_path": "base64_encoded_nifti_data"
}
```

**Response**:
```json
{
  "diagnosis_text": "Based on the cardiac imaging analysis...",
  "links": ["https://reference1.com", "https://reference2.com"],
  "confidence": 0.87,
  "num_hits": 3
}
```

### Search Endpoint Details

**POST `/search`**

```json
{
  "query_text": "patient with ventricular septal defect"
}
```

## 🔬 Technical Deep Dive

### Embedding Space Analysis

The 512-dimensional embedding space captures:

1. **Anatomical Structure**: Chamber sizes, wall thickness, vessel diameters
2. **Pathological Patterns**: Abnormal connections, dilations, malformations
3. **Spatial Relationships**: Relative positioning of cardiac structures
4. **Intensity Characteristics**: Tissue contrast and enhancement patterns

### Model Performance

**3D UNet Segmentation**:
- Training Accuracy: ~85% pixel-wise
- Validation Loss: CrossEntropy ~0.3
- Inference Time: ~2-3 seconds per volume (GPU)

**Embedding Quality**:
- Dimensionality: 512-D
- Distance Metric: Cosine similarity (angular)
- Retrieval Precision@3: High similarity for cardiac conditions

### Data Pipeline

```
NIfTI Input (Variable size)
    ↓
Cubic Padding + Resize (128³)
    ↓
3D CNN Feature Extraction
    ↓
Transformer Attention Processing  
    ↓
512-D Embedding Vector
    ↓
Vespa Vector Database Storage
    ↓
ANN Similarity Search
    ↓
Clinical Record Retrieval
    ↓
LLM Diagnosis Generation
```

## 🌐 Web Interface

**Location**: `web/`

Next.js application providing:
- **File Upload**: Drag-and-drop NIfTI upload interface
- **3D Visualization**: Interactive heart region exploration
- **Diagnosis Dashboard**: AI-generated reports with citations
- **Comparison Tools**: Side-by-side cardiac structure analysis

**Key Components**:
- `ComparisonViewer.tsx`: Multi-patient comparison interface
- `Heart.tsx`: 3D WebGL heart rendering
- `ProcessingState.tsx`: Real-time analysis progress
- `ResultsDashboard.tsx`: Comprehensive diagnosis presentation

## 🧪 Development & Training

### Training New Models

```python
# See ml/embeder.ipynb for complete training pipeline

# Data preparation
dataset_transformed = [(preprocess_3d_tensor(t[0]), preprocess_layers(t[1])) 
                      for t in dataset]

# Model training
model = UNet3D(in_channels=1, num_classes=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

# Training loop with accuracy tracking
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Forward pass, loss calculation, backprop
        ...
```

### Adding New Cardiac Conditions

1. **Update Schema**: Add new boolean fields to `medical_records.sd`
2. **Modify Smart Diagnosis**: Update condition list in `smart_diagnosis.py`
3. **Retrain Models**: Include new labeled data in training pipeline
4. **Update Visualization**: Add condition-specific rendering in `vis.py`

## 📊 Data Sources

The platform processes cardiac MRI data from:
- **HVSMR Dataset**: Congenital heart disease imaging
- **NIfTI Format**: Standardized medical imaging format
- **Clinical Metadata**: Patient demographics and condition labels
- **Segmentation Masks**: Expert-annotated cardiac region labels

Expected data structure:
```
data/
├── hvsmr_clinical.csv          # Patient metadata
└── 25226366/cropped/
    ├── pat0_cropped.nii.gz     # Original MRI
    ├── pat0_cropped_seg.nii.gz # Segmentation
    └── pat0_cropped_seg_endpoints.nii.gz
```

## 🤖 AI Integration

### Perplexity Integration

**Location**: `api2/smart_diagnosis.py`

The platform uses Perplexity's sonar-pro model for evidence-based diagnosis:

```python
def sMaRTDiagnosis(vespa_output: str):
    # Converts Vespa search results to clinical condition analysis
    # Returns diagnosis with medical literature citations
    # Handles 45+ cardiac conditions and abnormalities
```

**Capabilities**:
- Evidence-based diagnosis from medical literature
- Condition probability assessment
- Treatment recommendation suggestions
- Medical terminology simplification
- Citation tracking for clinical references

### Supported Conditions

The system can identify and analyze:

**Structural Abnormalities**:
- Ventricular Septal Defect (VSD)
- Atrial Septal Defect (ASD)  
- Double Outlet Right Ventricle (DORV)
- Transposition of Great Arteries (TGA)

**Cardiac Malpositions**:
- Dextrocardia, Mesocardia
- Inverted ventricles/atria
- Heterotaxy syndrome

**Vessel Abnormalities**:
- Bilateral Superior Vena Cava
- Tortuous vessels
- Pulmonary artery atresia

**Surgical Interventions**:
- Arterial/Atrial switch procedures
- Glenn/Fontan operations
- Rastelli procedure

## 📈 Performance & Scaling

### Computational Requirements

**Training**:
- GPU: NVIDIA RTX 3080+ (12GB+ VRAM)
- RAM: 32GB+ recommended
- Storage: 100GB+ for full dataset

**Inference**:
- GPU: GTX 1660+ (6GB+ VRAM)
- RAM: 16GB+ 
- CPU: Multi-core for Vespa operations

### Optimization Strategies

1. **Model Optimization**: 
   - Mixed precision training (FP16)
   - Gradient checkpointing for memory efficiency
   - Model quantization for deployment

2. **Data Pipeline**:
   - Asynchronous data loading
   - Memory-mapped file access
   - Batch processing for embeddings

3. **Vector Search**:
   - HNSW indexing in Vespa
   - Embedding dimension reduction (PCA/t-SNE)
   - Caching for frequent queries

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```python
# Reduce batch size or use gradient accumulation
torch.cuda.empty_cache()
# Add to training loop
```

**Vespa Connection Errors**:
```bash
# Check Vespa status
docker ps | grep vespa
# Restart if needed
docker restart vespa
```

**NIfTI Loading Issues**:
```python
# Verify file integrity
import nibabel as nib
img = nib.load("file.nii.gz")
print(f"Shape: {img.shape}, Dtype: {img.get_data_dtype()}")
```

**Low Segmentation Quality**:
- Verify input preprocessing matches training data
- Check for intensity normalization issues
- Ensure proper 3D orientation alignment

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Add type hints for new functions
- Include docstrings for ML models
- Test with sample NIfTI data
- Update README for new features

## 📧 Contact

**Project Lead**: Ethan Tam  
**License**: MIT  
**Repository**: [GitHub](https://github.com/your-username/hearti)

---

*Built with ❤️ for advancing cardiac healthcare through AI*
