# Industrial Defect Detection Application

A sophisticated Flask-based web application leveraging YOLO (You Only Look Once) models for real-time detection and analysis of industrial defects. This application is designed to enhance quality control processes in manufacturing environments.

## 🎯 Key Features

### 1. Gap Measurement Detection
- Precise measurement of gaps between components
- Ideal for assembly line quality control
- Ensures proper spacing and alignment in manufacturing

### 2. Weld Defect Detection
- Identifies common welding defects:
  - Porosity
  - Cracks
  - Incomplete fusion
  - Undercut
  - Slag inclusion
- Real-time analysis of weld quality
- Helps maintain welding standards and quality

### 3. Paint Aesthetics Analysis
- Detects paint-related issues:
  - Oil marks
  - Paint runs
  - Uneven coating
  - Surface imperfections
- Ensures consistent paint quality
- Identifies surface preparation issues

### 4. Video Processing
- Process video feeds for continuous monitoring
- Frame-by-frame analysis
- Timestamp-based defect tracking

### 5. AI-Powered Explanations
- Detailed explanations of detected defects
- Root cause analysis
- Preventive measures and recommendations
- Powered by GROQ LLM API

## 🏭 Industrial Use Cases

### Manufacturing Quality Control
- Assembly line inspection
- Automated defect detection
- Real-time quality monitoring
- Reduce manual inspection time

### Welding Operations
- Weld quality assurance
- Immediate feedback on weld defects
- Maintain consistent welding standards
- Reduce rework and waste

### Paint Shops
- Paint quality monitoring
- Surface preparation verification
- Coating uniformity check
- Reduce aesthetic defects

### General Applications
- Training new quality control personnel
- Documentation of defects
- Quality trend analysis
- Process improvement insights

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Adequate storage for model files
- GROQ API access

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/SaiPavankumar22/Industrial-Defect-Detector.git
   cd Industrial-Defect-Detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your GROQ API key:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

4. Download required model files:
   - `gap_final.pt` - For gap measurements
   - `weld_yolov8m-obb.pt` - For weld defect detection
   - `paint_best.pt` - For paint aesthetics analysis

5. Run the application:
   ```bash
   python app.py
   ```

## 📱 Usage Guide

1. **Home Page**
   - Access the application at `http://localhost:5000`
   - Choose between image or video analysis

2. **Image Analysis**
   - Upload an image
   - Select detection type:
     - Gap Measurement
     - Weld Defect
     - Paint Aesthetics
   - View results and AI-generated explanations

3. **Video Analysis**
   - Upload a video file
   - View frame-by-frame analysis
   - Check timestamps for defects

4. **Results**
   - View annotated images/frames
   - Read defect explanations
   - Get preventive recommendations

## 🛠️ Technical Details

- **Backend**: Flask (Python)
- **ML Models**: YOLOv8
- **LLM Integration**: GROQ API
- **Video Processing**: OpenCV
- **Frontend**: HTML/CSS/JavaScript

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 for object detection
- GROQ for LLM capabilities
- Flask for web framework
- OpenCV for image processing 