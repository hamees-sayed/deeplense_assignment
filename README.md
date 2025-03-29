### DeepLense GSoC Assignment 
Trained model checkpoints are available at:  [Google Drive - Model Checkpoints](https://drive.google.com/drive/folders/1nx47dQgX7yavo8oQ4e0-seyRaG1rDXY-?usp=sharing)  

## Common Test I. Multi-Class Classification

### Dataset and Classes  
The dataset consists of three types of lensing images:  

- **No Substructure**  
- **Sphere Substructure**  
- **Vortex Substructure**  

### Preprocessing Approach  
- Resized images to **256x256**, followed by center cropping to **224x224**.  
- Applied **data augmentation** (training only), including random rotation (10 degrees) and horizontal flipping for better generalization.  
- **Normalization** to scale pixel values, ensuring consistency across images.  

### Model Architecture  
Two CNN-based architectures were experimented with for performance comparison:

1. **ResNet-34**  
2. **EfficientNet-B3**  

Both models were initialized with **pretrained ImageNet weights**, and the final fully connected layer was replaced to match the number of classes. Models were trained using **cross-entropy loss** and the **Adam optimizer**, with validation accuracy used for model selection.  

### Results  

#### Test Accuracy  
- **ResNet-34:** **96.39%** (0.9639)  
- **EfficientNet-B3:** **92.24%** (0.9224)  

#### AUC Scores  
| Model            | No Substructure | Sphere Substructure | Vortex Substructure |
|-----------------|----------------|----------------------|----------------------|
| **ResNet-34**   | 0.99           | 0.99                 | 1.00                 |
| **EfficientNet-B3** | 0.99      | 0.98                 | 0.99                 |

---  

## Specific Test IV. Diffusion Models 

### Findings  
The results are measured based on the following parameters:

- `dtype`: `float16`
- `torch.compile` with mode `reduce-overhead`.
- ODE Solver used is `Euler` with `50` timesteps.   

| Model           | FID  | ODE Timesteps | Latency | Batch Size |
|---------------|------|---------------|---------|------------|
| **Transformer** | 27.6961 | 50          | 109ms   | 1          |
| **UNet**      | 34.8437 | 50            | 123ms   | 1          |

### Model Architecture  
A Flow Matching model was implemented and trained using two different backbones for performance comparison:

1. **UNet Backbone**  
   - Utilizes a ResNet-34 encoder with pretrained ImageNet weights.  

2. **Transformer Backbone**  
   - Based on a Vision Transformer (ViT) architecture with pretrained ImageNet weights.  
