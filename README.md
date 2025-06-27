# EchoFrame 

**A Multi-Modal AI Pipeline** that seamlessly integrates computer vision, natural language processing, and speech synthesis to transform static images into immersive audio stories.

EchoFrame demonstrates the power of multi-modal AI by combining multiple AI models in a sophisticated pipeline that goes from pixels to spoken narratives.

## Multi-Modal AI Architecture

EchoFrame showcases a complete **multi-modal AI pipeline** that processes visual, textual, and audio data:

```
Image Input →  Computer Vision →  Natural Language →  Audio Output
    ↓                  ↓                ↓                ↓
  PNG/JPG        Object Detection     Story Gen.       Speech
                 Pose Analysis        LLaMA 3.2        gTTS
                 Spatial Relations
```

## Key Features

- **Computer Vision Pipeline**: YOLOv8 object detection + MediaPipe pose estimation
- **Spatial Intelligence**: Advanced relationship detection between detected objects
- **Natural Language Generation**: LLaMA 3.2 integration for creative storytelling
- **Speech Synthesis**: Google Text-to-Speech for audio narration
- **Real-time Processing**: Optimized pipeline for efficient multi-modal inference

## Technical Stack

- **Computer Vision**: YOLOv8x (Object Detection) + MediaPipe (Pose Estimation)
- **Large Language Model**: LLaMA 3.2 via Ollama
- **Speech Synthesis**: Google Text-to-Speech (gTTS)
- **Image Processing**: OpenCV
- **Multi-Modal Orchestration**: Custom Python pipeline

## Requirements

- Python 3.8+
- YOLOv8 model file (`yolov8x.pt`)
- Ollama with LLaMA 3.2 model installed

## Installation

1. Clone this repository:
```bash
git clone https://github.com/srimadhavn/EchoFrame.git
cd EchoFrame
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install Ollama and pull LLaMA 3.2:
```bash
ollama pull llama3.2
```

## 🔄 Multi-Modal Workflow

EchoFrame implements a sophisticated **4-stage multi-modal pipeline**:

### Stage 1: Visual Analysis 
- **Object Detection**: YOLOv8 identifies and localizes objects with confidence scores
- **Pose Estimation**: MediaPipe analyzes human body poses and postures
- **Spatial Reasoning**: Custom algorithms detect relationships between objects

### Stage 2: Scene Understanding 
- **Context Fusion**: Combines visual detections into structured scene representation
- **Relationship Mapping**: Builds spatial and semantic connections between elements
- **Confidence Filtering**: Ensures high-quality inputs for downstream processing

### Stage 3: Language Generation 
- **Prompt Engineering**: Converts visual analysis into structured prompts
- **LLM Inference**: LLaMA 3.2 generates creative narratives from scene understanding
- **Context-Aware Storytelling**: Creates coherent stories that reflect visual content

### Stage 4: Audio Synthesis 
- **Text Processing**: Prepares generated text for speech synthesis
- **Voice Generation**: gTTS converts text to natural-sounding speech
- **Multi-Format Output**: Saves both text and audio versions

## Usage

Run the application with an image:

```bash
python main.py --image inputs/image.png
```

##  Multi-Modal Output

The application demonstrates end-to-end multi-modal processing by generating:
- **Visual**: `output/annotated.jpg` - Annotated image with AI detections
- **Textual**: `output/story.txt` - AI-generated narrative story  
- **Audio**: `output/story.mp3` - Synthesized speech narration

**Example Output Flow:**
```
Input: image.jpg
↓
Visual AI: "2 people, 1 dog, standing pose, outdoor setting"
↓  
Language AI: "On a sunny afternoon, two friends decided to take their..."
↓
Audio AI:  [Natural speech narration of the story]
```

## Technical Highlights

- **Multi-Modal Integration**: Seamless orchestration of 4 different AI models
- **Efficient Pipeline**: Optimized processing chain from pixels to audio
- **Intelligent Fusion**: Smart combination of computer vision and NLP insights  
- **Production Ready**: Robust error handling and modular architecture
- **Confidence-Based Processing**: Quality gates ensure reliable multi-modal outputs

## Why Multi-Modal AI Matters

EchoFrame demonstrates the future of AI applications - moving beyond single-mode processing to create richer, more human-like understanding. By combining:
- **Visual Intelligence** 
- **Spatial Reasoning**  
- **Creative Language** 
- **Natural Speech** 

