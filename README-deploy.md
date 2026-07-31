# Cloud deployment notes

This app is a Streamlit application and can be deployed to services such as Render, Railway, or Hugging Face Spaces.

## Recommended deployment

Use Render or Railway for a simple public deployment.

### Render
1. Create a new Web Service.
2. Connect this repository.
3. Set the build command to: `pip install -r requirements.txt`
4. Set the start command to: `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`
5. Add environment variable `PYTHON_VERSION=3.11`.

### Railway
1. Create a new project from the repository.
2. Use the existing Python environment.
3. Set the start command to: `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

### Hugging Face Spaces
1. Create a new Space with Streamlit SDK.
2. Upload this repository contents.
3. The app entry point should be `app.py`.
