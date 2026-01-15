# Video Processing Plugin for Dataiku DSS

A Dataiku DSS plugin that enables AI agents to analyze video content using multimodal Large Language Models (LLMs).

## Features

- **Watch Video Tool**: An agent tool that extracts frames from videos and analyzes them using vision-capable LLMs
- Intelligent frame sampling (up to 10 evenly-spaced frames)
- Automatic image optimization for efficient LLM processing
- Seamless integration with Dataiku's agent framework

## Requirements

- Dataiku DSS
- A multimodal LLM configured in Dataiku (e.g., GPT-4 Vision, Claude 3)
- Python 3.6+

### Python Dependencies

- `opencv-python` - Video frame extraction
- `langchain[all]` - LLM integration

## Installation

1. Download or clone this repository
2. In Dataiku DSS, go to **Plugins > Add Plugin > Development > Load from folder**
3. Select this plugin directory
4. Create the plugin's code environment when prompted

## Configuration

### Watch Video Tool

Configure the tool with the following parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_folder` | Folder | Dataiku managed folder containing your video files |
| `llm_id` | LLM | A multimodal LLM connection (must support vision) |

## Usage

The Watch Video tool is designed to be used by Dataiku agents. When invoked, it accepts:

- `video_name`: The filename of the video to analyze (e.g., `my_video.mp4`)
- `question`: A specific question about the video content

### Example

**Input:**
```json
{
  "video_name": "product_demo.mp4",
  "question": "What are the main features shown in this demo?"
}
```

**Output:**
```json
{
  "output": "Visual Analysis of 'product_demo.mp4': The demo shows...",
  "sources": []
}
```

## How It Works

1. **Video Validation**: Checks that the requested video exists in the configured folder
2. **Frame Extraction**: Downloads the video and extracts up to 10 evenly-spaced frames using OpenCV
3. **Image Optimization**: Resizes frames to 512px width to reduce token consumption
4. **LLM Analysis**: Sends frames with the user's question to a multimodal LLM
5. **Response**: Returns the LLM's visual analysis

## Project Structure

```
dss-plugin-video-tool/
├── plugin.json                    # Plugin configuration
├── python-agent-tools/
│   └── watch-video/
│       ├── tool.json             # Tool configuration
│       └── tool.py               # Tool implementation
├── python-lib/
│   └── videoprocessing/          # Shared library
└── code-env/python/
    └── spec/requirements.txt     # Dependencies
```

