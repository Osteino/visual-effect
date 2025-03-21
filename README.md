# visual-effect

A deep learning-based image effect generator that creates dream-like visuals/psychedelic versions of input images using deep dream techniques and PyTorch.

## Features

- Creates dream-like visuals/psychedelic effects from any input image
- Uses pre-trained ResNet18 model for feature extraction
- Customizable effect intensity and iterations
- GPU acceleration support (if available)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Pillow
- OpenCV
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Osteino/visual-effect
cd visual-effect
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your input image in the project directory as `input.jpg`

2. Run the script:
```bash
python main.py
```

3. The transformed image will be saved as `output_visual_effect.jpg`

## How It Works

The project uses deep dream techniques to enhance and exaggerate features that the neural network detects in the input image:

1. Loads a pre-trained ResNet18 model
2. Processes the input image through multiple layers
3. Optimizes the input to maximize feature activations
4. Applies gradient ascent to create psychedelic effects

## Configuration

You can adjust these parameters in `main.py`:

- `num_iterations`: Controls the intensity of the effect (default: 100)
- `lr`: Learning rate for gradient ascent (default: 0.01)
- `size`: Input image size (default: 224x224)

## Project Structure

```
visual-effect/
├── main.py           # Main script
├── requirements.txt  # Dependencies
├── README.md        # Documentation
├── input.jpg        # Your input image
└── data/            # Model weights directory
```

## Dependencies

- numpy
- Pillow
- torch
- torchvision
- matplotlib
- opencv-python

## Known Issues

- Large images may require significant processing time
- GPU recommended for faster processing
- Memory usage can be high for complex images


## Contributing

Feel free to submit issues and enhancement requests!

1. Fork the repo
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request