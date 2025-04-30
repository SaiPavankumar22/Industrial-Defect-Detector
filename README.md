# Defect Detection Application

A Flask-based web application for detecting various types of defects in components using YOLO models.

## Features

- Gap measurement detection
- Weld defect detection
- Paint aesthetics analysis
- Video processing capabilities
- Real-time defect analysis and explanation

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the required model files (not included in the repository due to size):
   - gap_final.pt
   - weld_yolov8m-obb.pt
   - paint_best.pt

4. Set up environment variables:
   - Create a `.env` file with your GROQ API key:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

5. Run the application:
   ```bash
   python app.py
   ```

## Project Structure

- `app.py` - Main Flask application
- `templates/` - HTML templates
- `uploads/` - Temporary storage for uploaded files
- `flask_session/` - Session data storage

## Note

The model files (`.pt` files) are not included in the repository due to their large size. You'll need to obtain these files separately and place them in the root directory of the project.

## License

[Add your license information here] 