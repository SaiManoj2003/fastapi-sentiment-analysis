from fastapi import FastAPI, UploadFile, File, HTTPException, Response, BackgroundTasks
from fastapi.responses import HTMLResponse
from starlette.responses import FileResponse
import os
import io
import logging
import matplotlib.pyplot as plt
import pandas as pd

from .model import analyze_audio, generate_plots, get_summary

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the directory to store uploaded files
UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save the file to the uploads directory
        with open(file_path, "wb") as f:
            file_content = await file.read()
            f.write(file_content)
        
        logger.info(f"File saved successfully at {file_path}")

        # Analyze the audio and generate the sentiment dataframe
        df = await analyze_audio(file_path)
        if df is None:
            raise HTTPException(status_code=500, detail="Error processing audio file.")
        
        # Generate plots from the sentiment dataframe
        plot_buffer = generate_plots(df)
        summary = get_summary(df)
        
        # Prepare the response
        response = {
            "summary": summary,
            "plot_url": f"/get-plot/{file.filename}"
        }
        
        # Optionally remove the file if you don't need it anymore
        # os.remove(file_path)
        # logger.info(f"File {file_path} removed successfully after processing.")

        return response

    except Exception as e:
        logger.error(f"Error uploading audio: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the audio file: {str(e)}")

@app.get("/get-plot/{filename}")
async def get_plot(filename: str):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Analyze the audio file and get the sentiment dataframe
        df = await analyze_audio(file_path)
        
        # Generate the plot
        plot_buffer = generate_plots(df)
        
        # Get the plot buffer content
        buf_contents = plot_buffer.getvalue()
        plot_buffer.close()

        headers = {'Content-Disposition': 'inline; filename="plot.png"'}
        return Response(content=buf_contents, headers=headers, media_type='image/png')

    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the plot: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

@app.get("/plot/")
async def get_plot_page():
    html_content = """
    <html>
        <body>
            <h2>Upload an audio file to analyze sentiment</h2>
            <form action="/upload-audio/" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)