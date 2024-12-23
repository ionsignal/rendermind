![RenderMind](logo.png)

# Rendermind Video Processing Pipeline

An efficient, GPU-accelerated framework for processing videos at the frame level using customizable pipelines, designed to leverage multiple GPUs for high performance.

## Overview

Rendermind provides a flexible and extensible architecture for video processing, allowing you to define custom pipelines composed of multiple "processors," such as upscalers, denoisers, and more. Each processor can be configured independently, and the pipeline can be distributed across multiple GPUs for faster processing.

## Key Features

- **Customizable Processing Pipelines**: Create complex processing sequences by chaining together multiple processors.
- **Multi-GPU Support**: Automatically distribute the workload across multiple GPUs for improved performance.
- **Scalable Tiling Mechanism**: Efficiently process high-resolution frames using tiling to manage GPU memory constraints.
- **Graceful Shutdown and Interrupt Handling**: Safely handle termination signals to prevent data loss and ensure proper resource cleanup.
- **Extensible Framework**: Easily add new processors by inheriting from base classes.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Configuring Pipelines](#configuring-pipelines)
  - [Using Multiple GPUs](#using-multiple-gpus)
- [Adding Custom Processors](#adding-custom-processors)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)
- [License](#license)

## Requirements

- **Python 3.10 or higher**
- **PyTorch**: For GPU computations.
- **ffmpeg**: For video handling.
- **CUDA-compatible GPUs**: For leveraging GPU acceleration.
- Additional Python packages as listed in `requirements.txt`.

## Installation (Linux)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rendermind.git
cd rendermind
```

### 2. Install Dependencies

Create a virtual environment (optional but recommended), and install dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install ffmpeg

Ensure `ffmpeg` is installed and available in your system PATH.

- **Ubuntu/Debian**:

  ```bash
  sudo apt-get update
  sudo apt-get install ffmpeg
  ```

- **Windows**:

  Download the static build from the [ffmpeg website](https://ffmpeg.org/download.html#build-windows) and add it to your PATH.

**Note: This application has not been tested on Windows or MacOS. (planned for future releases)**

### 4. Set Up Model Weights

Download the required model weights and place them in the appropriate directory. For example, for the `SpandrelProcessor`:

- Create a `weights` directory in the project root.

  ```bash
  mkdir weights
  ```

- Place your `.safetensors` model files in the `weights` directory.

### 5. Include the Spandrel Library

The Spandrel library is required for certain processors like `SpandrelProcessor`.

## Project Structure

```
rendermind/
├── src/
│   ├── main.py                             # Command-line interface entry point (for testing/dev).
│   ├── api/                                # Web API layer (Placeholder - not implemented in provided code)
│   │   ├── app.py                          # Flask/FastAPI application setup.
│   │   └── routes/                         # API endpoint definitions.
│   ├── rendermind/                         # Core logic of the application
│   │   ├── transformation_manager.py       # Manages the video transformation workflow.
│   │   ├── transformation/                 # Contains transformation-related classes
│   │   │   ├── gpu_pool.py                 # Manages GPU workers for transformations.
│   │   │   ├── transformation_pipeline.py  # Defines the processing pipeline for transformations.
│   │   │   └── processors/                 # Implementations of different image processing transforms
│   │   │       ├── spandrel.py             # Implements a processor using the Spandrel model library.
│   │   │       └── base.py                 # Abstract base class for all processors.
│   │   ├── utils/                          # Utility functions
│   │   │   ├── tensor_processor.py         # Implements tensor processing functions like tiled scaling and gaussian blur.
│   │   │   ├── timing.py                   # Provides timing utilities for measuring execution time.
│   │   │   ├── stats.py                    # Formats processing statistics into a human-readable string.
│   │   │   └── gpu.py                      # Provides GPU related utility functions.
├── weights/
│   └── [model files]
├── requirements.txt
└── setup.py
```

## Usage

### Quick Start

Below is a minimal example to process a video using the default settings.

```python
from rendermind.video_processor import VideoProcessor

# Initialize the VideoProcessor with default configuration
processor = VideoProcessor()

# Paths to your input and output video files
input_video = 'input_video.mp4'
output_video = 'output_video.mp4'

# Process the video
processor.process_video(input_video, output_video)
```

This will use the default processing pipeline, which includes the `SpandrelProcessor` with default settings.

### Configuring Pipelines

To customize the processing pipeline, provide a `pipeline_config` when initializing `VideoProcessor`.

```python
pipeline_config = [
    {
        'type': 'spandrel',
        'name': '4x_upscaler',
        'model_path': 'weights/4xRealWebPhoto-v4-dat2.safetensors',
        'config': {
            'tile_size': 512,
            'min_tile_size': 128,
            'overlap': 32
        }
    },
    # Add more processors as needed
]

# Specify devices (e.g., "cuda:0" or "cuda:0,cuda:1")
devices = "cuda:0"

processor = VideoProcessor(devices=devices, pipeline_config=pipeline_config)
```

### Using Multiple GPUs

To leverage multiple GPUs, list them in the `devices` parameter.

```python
devices = "cuda:0,cuda:1"

processor = VideoProcessor(devices=devices, pipeline_config=pipeline_config)
```

The `GPUWorkerPool` will automatically distribute frames among the specified GPUs.

## Adding Custom Processors

To extend the processing capabilities, you can create custom processors.

## Performance Tips

- **Optimize Tile Sizes**: Adjust `tile_size`, `min_tile_size`, and `overlap` in the processor configuration to balance between performance and memory usage.
- **Monitor GPU Memory**: Use tools like `nvidia-smi` to monitor GPU memory usage and adjust configurations accordingly.
- **Use Appropriate Batch Sizes**: While the provided code processes frames individually, batch processing can be implemented for models that support it.
- **Warm-Up Models**: The pipeline includes a warm-up step to pre-load models onto the GPU, reducing initialization overhead during processing.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

1. **Fork** the repository.
2. **Create** a new branch with a descriptive name.
3. **Make** your changes.
4. **Submit** a pull request to the `main` branch.

Please ensure that your code follows the existing style and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
