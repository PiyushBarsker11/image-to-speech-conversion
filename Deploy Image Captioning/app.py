from flask import Flask, render_template, request
from caption import caption_and_speak_image  # Make sure the import is correct for your project structure

app = Flask(__name__)

# Route to display the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and caption generation
@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        img = request.files['image']
        if img:
            img_path = "static/" + img.filename
            img.save(img_path)

            # Call the caption and speech function to get the caption
            caption = caption_and_speak_image(img_path)

            # Create a dictionary with the image and caption for rendering
            result_dic = {
                'image': img_path,
                'description': caption  # Display the actual caption generated
            }
            return render_template('index.html', results=result_dic)

if __name__ == '__main__':
    app.run(debug=True)
