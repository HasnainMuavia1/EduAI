# EduAI

EduAI is a comprehensive AI-powered educational platform designed to enhance learning and productivity. It leverages advanced machine learning models and APIs to provide features like video transcription, document interaction, smart content generation, and Optical Character Recognition (OCR).

## 🚀 Features

### 1. 🎥 Lecture Transcription
*   **YouTube to Text**: Simply provide a YouTube URL to download and transcribe the audio.
*   **File Upload**: Upload audio or video files directly for transcription.
*   **Powered by**: IBM Watson Speech-to-Text for high-accuracy transcriptions.

### 2. 📚 Chat with Book
*   **Document Support**: Upload PDF, DOCX, or TXT files.
*   **Interactive Chat**: Ask questions regarding the content of your uploaded documents.
*   **Contextual Understanding**: Uses Groq (Llama 3) to analyze the text and provide accurate context-aware answers.

### 3. 📝 Generate Paper / Smart Chat
*   **Content Creation**: Generate formatted educational content such as:
    *   Multiple Choice Questions (MCQs)
    *   Structured answers with explanations
    *   Fill-in-the-blank exercises
*   **Custom Formatting**: Automatically formats responses with HTML for clean presentation.

### 4. 📷 OCR (Optical Character Recognition)
*   **Image to Text**: Upload images containing text to extract the content digitally.
*   **Advanced Vision**: Utilizes Llama Vision models via Groq to interpret and extract text from complex images.

---

## 🛠️ Technology Stack

*   **Backend**: Django (Python)
*   **AI Inference**: Groq API (Llama 3 models)
*   **Speech Processing**: IBM Watson Speech-to-Text
*   **Media Processing**: `yt-dlp` (YouTube download), `ffmpeg` (Audio conversion)
*   **Document Handling**: `PyPDF2`, `python-docx`

---

## ⚙️ Installation & Setup

### Prerequisites
*   Python 3.8+
*   **FFmpeg**: Required for audio conversion. Ensure `ffmpeg` is installed and added to your system's PATH.

### Steps

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd EduAI
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Configuration**
    Create a `.env` file in the root directory (same level as `manage.py`) and add your API credentials:

    ```env
    # IBM Watson Credentials
    SPEECH_TO_TEXT_IAM_APIKEY=your_ibm_api_key
    SPEECH_TO_TEXT_URL=your_ibm_service_url

    # Groq API Key
    GROQ_API_KEY=your_groq_api_key
    ```

5.  **Run Migrations**
    Initialize the database:
    ```bash
    python manage.py migrate
    ```

6.  **Start the Development Server**
    ```bash
    python manage.py runserver
    ```

7.  **Access the Application**
    Open your browser and navigate to `http://127.0.0.1:8000/`.

---

## 📂 Project Structure

*   **EduAI/**: Main project configuration.
*   **webapp/**: Core application logic.
    *   `views.py`: Handles file processing, API calls to Groq/Watson, and response formatting.
    *   `urls.py`: Application routing.
*   **templates/**: HTML templates for the frontend.
*   **media/**: Stores temporary uploaded files and converted audio (ensure this directory exists or permissions are set).

---

## ⚠️ Notes
*   Ensure you have a stable internet connection as the app relies on external APIs (Groq, IBM Watson).
*   API usage may incur costs depending on your provider plans.
