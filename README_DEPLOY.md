# Bird Sound Classifier Deployment

This is a Streamlit app. The deployable runtime needs:

- `app.py`
- `requirements.txt`
- `packages.txt` if the host supports apt packages
- `bird_sound_classifier_final/`

The model file is large, so use Git LFS before pushing to GitHub or Hugging Face:

```powershell
git init
git lfs install
git add .gitattributes .gitignore app.py requirements.txt packages.txt Dockerfile README_DEPLOY.md bird_sound_classifier_final
git commit -m "Prepare Streamlit deployment"
```

## Streamlit Community Cloud

1. Push the repo to GitHub with Git LFS enabled.
2. Create a new Streamlit app from that repo.
3. Set the main file path to `app.py`.

## Hugging Face Spaces

Create a new Streamlit Space, then push these files with Git LFS enabled. Hugging Face Spaces handles large model files well.

## Docker Hosts

For Render, Fly.io, Railway, or a Docker Hugging Face Space, deploy this repository using the included `Dockerfile`. The app listens on port `8501`.
