#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers diffusers torch torchvision torchaudio')
get_ipython().system('pip install flask flask-ngrok pyngrok nltk scikit-learn matplotlib pillow')
get_ipython().system('pip install ipywidgets')


# In[2]:


get_ipython().system('mkdir -p artvision/static/images artvision/static/styles artvision/templates')


# In[3]:


from google.colab import files
uploaded = files.upload()


# In[4]:


get_ipython().system('mv starry_night.jpg artvision/static/styles/')
get_ipython().system('mv monet.jpg artvision/static/styles/')
get_ipython().system('mv picasso.jpg artvision/static/styles/')
get_ipython().system('mv kandinsky.jpg artvision/static/styles/')


# In[5]:


get_ipython().system('ls artvision/static/styles/')


# In[6]:


get_ipython().run_cell_magic('writefile', 'artvision/emotion_analyzer.py', 'import nltk\nfrom nltk.tokenize import word_tokenize\nfrom transformers import pipeline\nimport numpy as np\n\n# Download necessary NLTK resources\nnltk.download(\'punkt_tab\')\nnltk.download(\'stopwords\')\n\nclass EmotionAnalyzer:\n    def __init__(self):\n        # Initialize emotion classification pipeline\n        self.emotion_classifier = pipeline("text-classification",\n                                          model="j-hartmann/emotion-english-distilroberta-base",\n                                          return_all_scores=True)\n\n        # Emotion to color/mood mapping\n        self.emotion_mappings = {\n            \'joy\': {\'color_boost\': [1.2, 1.1, 0.9], \'prompt_prefix\': \'bright, cheerful, vibrant\'},\n            \'sadness\': {\'color_boost\': [0.8, 0.8, 1.1], \'prompt_prefix\': \'melancholic, somber, muted\'},\n            \'anger\': {\'color_boost\': [1.3, 0.7, 0.7], \'prompt_prefix\': \'intense, fiery, dramatic\'},\n            \'fear\': {\'color_boost\': [0.7, 0.7, 0.9], \'prompt_prefix\': \'dark, ominous, shadowy\'},\n            \'surprise\': {\'color_boost\': [1.1, 1.1, 0.8], \'prompt_prefix\': \'vibrant, dynamic, unexpected\'},\n            \'disgust\': {\'color_boost\': [0.9, 1.1, 0.8], \'prompt_prefix\': \'unsettling, distorted, unpleasant\'},\n            \'neutral\': {\'color_boost\': [1.0, 1.0, 1.0], \'prompt_prefix\': \'balanced, neutral, clear\'}\n        }\n\n    def preprocess_text(self, text):\n        """Preprocess text by tokenizing and removing stopwords"""\n        tokens = word_tokenize(text.lower())\n        stop_words = set(nltk.corpus.stopwords.words(\'english\'))\n        filtered_tokens = [w for w in tokens if w not in stop_words]\n        return \' \'.join(filtered_tokens)\n\n    def analyze_emotion(self, text):\n        """Analyze the emotional content of text"""\n        # Preprocess the text\n        processed_text = self.preprocess_text(text)\n\n        # Get emotion scores\n        emotion_scores = self.emotion_classifier(text)[0]\n\n        # Find the dominant emotion\n        dominant_emotion = max(emotion_scores, key=lambda x: x[\'score\'])\n\n        # Get the mapping for the dominant emotion (default to neutral if not found)\n        emotion_name = dominant_emotion[\'label\']\n        mapping = self.emotion_mappings.get(emotion_name, self.emotion_mappings[\'neutral\'])\n\n        result = {\n            \'dominant_emotion\': emotion_name,\n            \'emotion_score\': dominant_emotion[\'score\'],\n            \'color_boost\': mapping[\'color_boost\'],\n            \'prompt_prefix\': mapping[\'prompt_prefix\'],\n            \'all_emotions\': {item[\'label\']: item[\'score\'] for item in emotion_scores}\n        }\n\n        return result\n')


# In[7]:


get_ipython().run_cell_magic('writefile', 'artvision/image_generator.py', 'import torch\nfrom diffusers import StableDiffusionPipeline\nfrom PIL import Image\nimport numpy as np\nimport os\n\nclass ImageGenerator:\n    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):\n        """Initialize the image generator with a specified model"""\n        self.device = "cuda" if torch.cuda.is_available() else "cpu"\n        print(f"Using device: {self.device}")\n\n        # Load the stable diffusion pipeline\n        self.pipe = StableDiffusionPipeline.from_pretrained(\n            model_id,\n            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,\n            safety_checker=None  # Disable safety checker for artistic freedom\n        )\n        self.pipe = self.pipe.to(self.device)\n\n        # Enable memory efficient attention if on CUDA\n        if self.device == "cuda":\n            self.pipe.enable_attention_slicing()\n\n        # Create directory for saving images if it doesn\'t exist\n        os.makedirs("artvision/static/images", exist_ok=True)\n\n    def generate_image(self, prompt, emotion_data, seed=None, guidance_scale=7.5, steps=30):\n        """Generate an image based on prompt and emotion data"""\n        # Enhance prompt with emotional context\n        enhanced_prompt = f"{emotion_data[\'prompt_prefix\']}, {prompt}"\n        print(f"Enhanced prompt: {enhanced_prompt}")\n\n        # Add negative prompt for better results\n        negative_prompt = "blurry, bad quality, distorted, deformed"\n\n        # Set random seed for reproducibility if provided\n        generator = None\n        if seed is not None:\n            generator = torch.Generator(device=self.device).manual_seed(seed)\n\n        # Generate the image\n        with torch.autocast(self.device):\n            image = self.pipe(\n                enhanced_prompt,\n                negative_prompt=negative_prompt,\n                guidance_scale=guidance_scale,\n                num_inference_steps=steps,\n                generator=generator\n            ).images[0]\n\n        # Apply color adjustments based on emotion\n        image_array = np.array(image)\n        color_boost = emotion_data[\'color_boost\']\n\n        # Apply color boost factors to RGB channels\n        adjusted_image = np.clip(image_array * np.array(color_boost), 0, 255).astype(np.uint8)\n\n        return Image.fromarray(adjusted_image)\n\n    def save_image(self, image, filename):\n        """Save the generated image to disk"""\n        filepath = os.path.join("artvision/static/images", filename)\n        image.save(filepath)\n        return filepath\n')


# In[8]:


get_ipython().run_cell_magic('writefile', 'artvision/style_transfer.py', 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport torchvision.transforms as transforms\nimport torchvision.models as models\nfrom PIL import Image\nimport numpy as np\nimport os\n\nclass ContentLoss(nn.Module):\n    def __init__(self, target):\n        super(ContentLoss, self).__init__()\n        self.target = target.detach()\n        self.loss = None\n\n    def forward(self, x):\n        self.loss = F.mse_loss(x, self.target)\n        return x\n\nclass StyleLoss(nn.Module):\n    def __init__(self, target_feature):\n        super(StyleLoss, self).__init__()\n        self.target = self.gram_matrix(target_feature).detach()\n        self.loss = None\n\n    def forward(self, x):\n        G = self.gram_matrix(x)\n        self.loss = F.mse_loss(G, self.target)\n        return x\n\n    def gram_matrix(self, input):\n        batch_size, n_channels, height, width = input.size()\n        features = input.view(batch_size * n_channels, height * width)\n        G = torch.mm(features, features.t())\n        return G.div(batch_size * n_channels * height * width)\n\nclass Normalization(nn.Module):\n    def __init__(self, mean, std):\n        super(Normalization, self).__init__()\n        self.mean = mean.clone().detach().view(-1, 1, 1)\n        self.std = std.clone().detach().view(-1, 1, 1)\n\n    def forward(self, img):\n        return (img - self.mean) / self.std\n\nclass StyleTransfer:\n    def __init__(self):\n        # Load pre-trained VGG19 model\n        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()\n\n        # Define normalization mean and std\n        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)\n        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)\n\n        # Content and style layers\n        self.content_layers = [\'conv_4\']\n        self.style_layers = [\'conv_1\', \'conv_2\', \'conv_3\', \'conv_4\', \'conv_5\']\n\n        # Directory for style images\n        self.style_dir = "artvision/static/styles"\n\n        # Default styles\n        self.default_styles = {\n            "Van Gogh": "artvision/static/styles/starry_night.jpg",\n            "Kandinsky": "artvision/static/styles/kandinsky.jpg",\n            "Monet": "artvision/static/styles/monet.jpg",\n            "Picasso": "artvision/static/styles/picasso.jpg"\n        }\n\n    def load_image(self, path, size=512):\n        """Load an image and convert it to a tensor"""\n        image = Image.open(path)\n        loader = transforms.Compose([\n            transforms.Resize(size),\n            transforms.CenterCrop(size),\n            transforms.ToTensor()\n        ])\n        image = loader(image).unsqueeze(0).to(self.device)\n        return image\n\n    def get_model_and_losses(self, content_img, style_img):\n        """Set up the model, losses, and feature extraction"""\n        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(self.device)\n\n        # Create a sequential model with content and style losses\n        model = nn.Sequential(normalization)\n        content_losses = []\n        style_losses = []\n\n        # Current position in the model\n        i = 0\n        for layer in self.cnn.children():\n            if isinstance(layer, nn.Conv2d):\n                i += 1\n                name = f\'conv_{i}\'\n            elif isinstance(layer, nn.ReLU):\n                name = f\'relu_{i}\'\n                layer = nn.ReLU(inplace=False)\n            elif isinstance(layer, nn.MaxPool2d):\n                name = f\'pool_{i}\'\n            elif isinstance(layer, nn.BatchNorm2d):\n                name = f\'bn_{i}\'\n            else:\n                raise RuntimeError(f\'Unrecognized layer: {layer.__class__.__name__}\')\n\n            model.add_module(name, layer)\n\n            # Add content loss\n            if name in self.content_layers:\n                target = model(content_img).detach()\n                content_loss = ContentLoss(target)\n                model.add_module(f"content_loss_{i}", content_loss)\n                content_losses.append(content_loss)\n\n            # Add style loss\n            if name in self.style_layers:\n                target_feature = model(style_img).detach()\n                style_loss = StyleLoss(target_feature)\n                model.add_module(f"style_loss_{i}", style_loss)\n                style_losses.append(style_loss)\n\n        # Trim off the layers after the last content and style losses\n        for i in range(len(model) - 1, -1, -1):\n            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):\n                break\n\n        model = model[:(i + 1)]\n\n        return model, style_losses, content_losses\n\n    def apply_style(self, content_img_path, style_name, output_path, num_steps=100,\n                   style_weight=1000000, content_weight=1):\n        """Apply a style to a content image"""\n        # Load content image\n        content_img = self.load_image(content_img_path)\n\n        # Load style image\n        if style_name in self.default_styles:\n            style_img_path = self.default_styles[style_name]\n        else:\n            style_img_path = os.path.join(self.style_dir, f"{style_name}.jpg")\n\n        style_img = self.load_image(style_img_path)\n\n        # Create input image (content image clone)\n        input_img = content_img.clone()\n\n        # Set up the optimizer\n        optimizer = torch.optim.LBFGS([input_img.requires_grad_()])\n\n        # Get model and losses\n        model, style_losses, content_losses = self.get_model_and_losses(content_img, style_img)\n\n        # Run the optimization\n        run = [0]\n        while run[0] <= num_steps:\n            def closure():\n                input_img.data.clamp_(0, 1)\n\n                optimizer.zero_grad()\n                model(input_img)\n\n                style_score = 0\n                content_score = 0\n\n                for sl in style_losses:\n                    style_score += sl.loss\n                for cl in content_losses:\n                    content_score += cl.loss\n\n                style_score *= style_weight\n                content_score *= content_weight\n\n                loss = style_score + content_score\n                loss.backward()\n\n                run[0] += 1\n                if run[0] % 20 == 0:\n                    print(f"Run {run[0]}/{num_steps}")\n\n                return style_score + content_score\n\n            optimizer.step(closure)\n\n        # Denormalize the output image\n        input_img.data.clamp_(0, 1)\n\n        # Convert tensor to PIL image\n        unloader = transforms.ToPILImage()\n        output_img = input_img[0].cpu().clone()\n        output_img = unloader(output_img)\n\n        # Save the output image\n        output_img.save(output_path)\n\n        return output_path\n')


# In[9]:


get_ipython().run_cell_magic('writefile', 'artvision/templates/index.html', '<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>ArtVision - Emotion-Driven Text-to-Image Generation</title>\n    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">\n    <style>\n        body {\n            background-color: #f8f9fa;\n            font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif;\n        }\n        .header {\n            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);\n            color: white;\n            padding: 2rem 0;\n            margin-bottom: 2rem;\n            border-radius: 0 0 10px 10px;\n        }\n        .card {\n            border-radius: 15px;\n            overflow: hidden;\n            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);\n            margin-bottom: 2rem;\n            transition: transform 0.3s ease;\n        }\n        .card:hover {\n            transform: translateY(-5px);\n        }\n        .btn-primary {\n            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);\n            border: none;\n            padding: 10px 20px;\n            font-weight: bold;\n        }\n        .btn-primary:hover {\n            background: linear-gradient(135deg, #5a0cb0 0%, #1565e0 100%);\n        }\n        .image-container {\n            position: relative;\n            overflow: hidden;\n            border-radius: 10px;\n            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);\n        }\n        .image-container img {\n            width: 100%;\n            transition: transform 0.5s ease;\n        }\n        .image-container:hover img {\n            transform: scale(1.03);\n        }\n        .emotion-badge {\n            position: absolute;\n            top: 10px;\n            right: 10px;\n            background: rgba(255, 255, 255, 0.9);\n            padding: 5px 10px;\n            border-radius: 20px;\n            font-weight: bold;\n            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);\n        }\n        .gallery {\n            display: grid;\n            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));\n            gap: 20px;\n            margin-top: 2rem;\n        }\n        .loading {\n            display: none;\n            text-align: center;\n            margin: 20px 0;\n        }\n        .spinner-border {\n            width: 3rem;\n            height: 3rem;\n        }\n    </style>\n</head>\n<body>\n    <div class="header text-center">\n        <h1 class="display-4">ArtVision</h1>\n        <p class="lead">Emotion-Driven Text-to-Image Generation with Style Transfer</p>\n    </div>\n\n    <div class="container">\n        <div class="row">\n            <div class="col-lg-8">\n')


# In[10]:


get_ipython().run_cell_magic('writefile', 'artvision/demo.py', 'import os\nimport uuid\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, HTML, clear_output\nimport ipywidgets as widgets\nfrom emotion_analyzer import EmotionAnalyzer\nfrom image_generator import ImageGenerator\nfrom style_transfer import StyleTransfer\n\nclass ArtVisionDemo:\n    def __init__(self):\n        self.emotion_analyzer = EmotionAnalyzer()\n        self.image_generator = ImageGenerator()\n        self.style_transfer = StyleTransfer()\n\n        # Available styles\n        self.styles = {\n            "none": "No Style",\n            "Van Gogh": "Starry Night",\n            "Kandinsky": "Kandinsky",\n            "Monet": "Monet",\n            "Picasso": "Picasso"\n        }\n\n        # Create output directory\n        os.makedirs("artvision/static/images", exist_ok=True)\n\n    def generate_image(self, prompt, style_name="none", seed=None):\n        """Generate an image based on the prompt and apply style if selected"""\n        # Analyze emotion in the text\n        emotion_data = self.emotion_analyzer.analyze_emotion(prompt)\n\n        # Display emotion analysis\n        print(f"Detected emotion: {emotion_data[\'dominant_emotion\']} (confidence: {emotion_data[\'emotion_score\']:.2f})")\n        print(f"Emotional prompt enhancement: {emotion_data[\'prompt_prefix\']}")\n\n        # Generate a unique ID for this generation\n        generation_id = str(uuid.uuid4())[:8]\n\n        # Generate the base image\n        print("Generating base image...")\n        base_image = self.image_generator.generate_image(prompt, emotion_data, seed)\n        base_image_path = self.image_generator.save_image(base_image, f"{generation_id}_base.jpg")\n\n        # Display the base image\n        plt.figure(figsize=(10, 10))\n        plt.imshow(base_image)\n        plt.axis(\'off\')\n        plt.title(f"Base Image: {prompt}")\n        plt.show()\n\n        # Apply style transfer if a style is selected\n        if style_name != "none":\n            print(f"Applying {self.styles[style_name]} style...")\n            final_image_path = os.path.join("artvision/static/images", f"{generation_id}_styled.jpg")\n            self.style_transfer.apply_style(\n                base_image_path,\n                style_name,\n                final_image_path,\n                num_steps=100  # Reduced for demo purposes\n            )\n\n            # Display the styled image\n            styled_image = plt.imread(final_image_path)\n            plt.figure(figsize=(10, 10))\n            plt.imshow(styled_image)\n            plt.axis(\'off\')\n            plt.title(f"Styled Image: {self.styles[style_name]}")\n            plt.show()\n\n            return final_image_path\n\n        return base_image_path\n')


# In[11]:


import sys
sys.path.append('/content/artvision')
get_ipython().system('ls /content/artvision/')


# In[12]:


from emotion_analyzer import EmotionAnalyzer
from image_generator import ImageGenerator
from style_transfer import StyleTransfer


# In[13]:


get_ipython().system('pip install gradio')


# In[14]:


from artvision.demo import ArtVisionDemo
demo = ArtVisionDemo()


# In[16]:


import gradio as gr

# Define the function to generate the image
def generate_image(prompt, style, seed):
    return demo.generate_image(prompt, style, seed)

# Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            # value='A peaceful mountain landscape at sunset',
            placeholder='Enter your text prompt...',
            label='Prompt'
        ),
        gr.Dropdown(
            choices=list(demo.styles.keys()),
            value='none',
            label='Style'
        ),
        gr.Slider(
            minimum=0,
            maximum=1000,
            step=1,
            label='Seed (Optional)'
        )
    ],
    outputs=gr.Image(),
    title="ArtVision: Emotion-Driven Text-to-Image Generation",
    description="Enter a text prompt, select a style and click submit to generate image..."
)

# Launch the interface
interface.launch()

