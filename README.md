# ArtVision 🎨

A web application hosted on Gradio that converts text to image with a chosen style while classifying the emotions displayed in the text prompt.
This project was originally built in Google Colab due to missing hardware infrastructure pertinent to this project.
We tried our best to host it forever but to no avail.

However, you can replicate this project by connecting to the T4 GPU on Google Colab and input code mentioned under "Cells" folder according to their ordinal number subsequently in each Google Colab Cell.

## 🚀 How to Run

This Project is deployed using:
- [Colab](https://colab.research.google.com/)
- [Gradio](https://www.gradio.app/)

After running all the cells on Colab, a Gradio link will be generated where Text to Image conversion can be implemented. Here, a text prompt along with the Style and Seed can be given as input to generate an image in that style along with the emotion detected in the text prompt.

## 📂 Files
- `Cells` - Folder containing code to be input subsequently in each Colab Cell.
- `Images` - Folder containing Style Images.
- `requirements.txt` - File containing all the dependencies required to run this web app.

## 💻 Demo Notebook
- [ArtVision: Emotion-Driven Text-to-Image Generation](https://colab.research.google.com/drive/1NcRID-OOglDyYwvXvLFbVy1zdZewm7JU?usp=sharing)
