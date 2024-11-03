from PIL import Image, ImageDraw, ImageFont
import markdown
from bs4 import BeautifulSoup
import requests
import glob
import os
from io import BytesIO


def download_image(url):
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    except:
        return None


def measure_text(text, font):
    return font.getlength(text)


def process_markdown(text):
    # Convert Markdown to HTML
    html = markdown.markdown(text)
    soup = BeautifulSoup(html, 'html.parser')
    return soup


def render_element(draw, element, x, y, width, fonts):
    current_y = y
    line_height = 24

    if element.name == 'h1':
        draw.text((x, current_y), element.text, font=fonts['h1'], fill="black")
        current_y += 40
    elif element.name == 'h2':
        draw.text((x, current_y), element.text, font=fonts['h2'], fill="black")
        current_y += 32
    elif element.name == 'p':
        words = element.text.split()
        line = []
        line_width = 0

        for word in words:
            word_width = measure_text(word + " ", fonts['p'])
            if line_width + word_width > width - 40:
                draw.text((x, current_y), " ".join(line), font=fonts['p'], fill="black")
                current_y += line_height
                line = [word]
                line_width = word_width
            else:
                line.append(word)
                line_width += word_width

        if line:
            draw.text((x, current_y), " ".join(line), font=fonts['p'], fill="black")
            current_y += line_height
    elif element.name == 'img':
        img_src = element.get('src', '')
        if img_src.startswith('http'):
            img = download_image(img_src)
            if img:
                # Scale image to fit width while maintaining aspect ratio
                aspect_ratio = img.height / img.width
                new_width = min(width - 40, img.width)
                new_height = int(new_width * aspect_ratio)
                img = img.resize((new_width, new_height))
                image.paste(img, (x, current_y))
                current_y += new_height + 20

    return current_y


def markdown_to_image(input_file, output_file):
    # Load fonts
    fonts = {
        'h1': ImageFont.truetype("arial.ttf", 32) if os.path.exists("arial.ttf") else ImageFont.load_default(),
        'h2': ImageFont.truetype("arial.ttf", 24) if os.path.exists("arial.ttf") else ImageFont.load_default(),
        'p': ImageFont.truetype("arial.ttf", 16) if os.path.exists("arial.ttf") else ImageFont.load_default()
    }

    # Read and parse markdown
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    soup = process_markdown(text)

    # Calculate required image size
    width = 800
    height = 100  # Starting height, will be adjusted

    # Create temporary image to calculate final height
    temp_img = Image.new('RGB', (width, height), 'white')
    temp_draw = ImageDraw.Draw(temp_img)

    current_y = 20
    for element in soup.find_all(['h1', 'h2', 'p', 'img']):
        current_y = render_element(temp_draw, element, 20, current_y, width, fonts)

    # Create final image with calculated height
    height = current_y + 20
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # Render content
    current_y = 20
    for element in soup.find_all(['h1', 'h2', 'p', 'img']):
        current_y = render_element(draw, element, 20, current_y, width, fonts)

    image.save(output_file)


# Process all markdown files
md_files = sorted(glob.glob("*.md"), key=lambda x: (x != "REPORT.md", x))

for i, md_file in enumerate(md_files, start=1):
    output_file = f'report{i}.png'
    print(f"Converting {md_file} to {output_file}")
    markdown_to_image(md_file, output_file)