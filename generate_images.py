#!/usr/bin/env python3
"""
Generate geometric shape images for vision testing.
"""

from PIL import Image, ImageDraw
import os
import json
import random
from pathlib import Path

def create_geometric_shape(shape_type, size=(256, 256), color=None):
    """Create a simple geometric shape image."""
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    
    if color is None:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    margin = 20
    width, height = size
    
    if shape_type == 'circle':
        draw.ellipse([margin, margin, width-margin, height-margin], fill=color)
    elif shape_type == 'square':
        draw.rectangle([margin, margin, width-margin, height-margin], fill=color)
    elif shape_type == 'triangle':
        points = [
            (width//2, margin),
            (margin, height-margin),
            (width-margin, height-margin)
        ]
        draw.polygon(points, fill=color)
    elif shape_type == 'rectangle':
        draw.rectangle([margin, margin*2, width-margin, height-margin*2], fill=color)
    elif shape_type == 'diamond':
        points = [
            (width//2, margin),
            (width-margin, height//2),
            (width//2, height-margin),
            (margin, height//2)
        ]
        draw.polygon(points, fill=color)
    elif shape_type == 'star':
        # Simple 5-pointed star
        center_x, center_y = width//2, height//2
        outer_radius = min(width, height)//2 - margin
        inner_radius = outer_radius // 2
        
        points = []
        for i in range(10):
            angle = i * 3.14159 / 5
            if i % 2 == 0:
                radius = outer_radius
            else:
                radius = inner_radius
            x = center_x + radius * (1 if i < 5 else -1) * abs(0.5 - (i % 5) / 4)
            y = center_y + radius * (1 if (i // 2) % 2 == 0 else -1) * abs(0.5 - ((i+1) % 5) / 4)
            points.append((int(x), int(y)))
        
        # Simplified star shape
        points = [
            (width//2, margin),
            (width//2 + 20, height//2 - 20),
            (width-margin, height//2),
            (width//2 + 20, height//2 + 20),
            (width//2, height-margin),
            (width//2 - 20, height//2 + 20),
            (margin, height//2),
            (width//2 - 20, height//2 - 20)
        ]
        draw.polygon(points, fill=color)
    
    return image

def generate_vision_prompts():
    """Generate vision test prompts with images."""
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)
    
    shapes = ['circle', 'square', 'triangle', 'rectangle', 'diamond', 'star']
    colors = [
        ('red', (255, 0, 0)),
        ('blue', (0, 0, 255)),
        ('green', (0, 255, 0)),
        ('yellow', (255, 255, 0)),
        ('purple', (128, 0, 128)),
        ('orange', (255, 165, 0))
    ]
    
    vision_prompts = []
    
    # Single image tasks
    for i in range(20):
        shape = random.choice(shapes)
        color_name, color_rgb = random.choice(colors)
        
        image = create_geometric_shape(shape, color=color_rgb)
        filename = f"shape_{i+1}_{shape}_{color_name}.png"
        image.save(images_dir / filename)
        
        vision_prompts.append({
            "id": f"vision_single_{i+1}",
            "prompt": "Describe what you see in this image. What shape and color is it?",
            "type": "vision_single",
            "images": [str(images_dir / filename)],
            "input_tokens": 15,
            "expected_output_tokens": 30,
            "expected_answer": f"{color_name} {shape}"
        })
    
    # Multiple image tasks (2-5 images per prompt)
    multi_image_counts = [2, 3, 4, 5]
    for count in multi_image_counts:
        for batch in range(3):  # 3 batches per count
            images = []
            shapes_in_prompt = []
            
            for img_idx in range(count):
                shape = random.choice(shapes)
                color_name, color_rgb = random.choice(colors)
                shapes_in_prompt.append(f"{color_name} {shape}")
                
                image = create_geometric_shape(shape, color=color_rgb)
                filename = f"multi_{count}_{batch}_{img_idx}_{shape}_{color_name}.png"
                image.save(images_dir / filename)
                images.append(str(images_dir / filename))
            
            vision_prompts.append({
                "id": f"vision_multi_{count}_{batch+1}",
                "prompt": f"Describe all {count} shapes you see in these images. List the color and shape of each.",
                "type": f"vision_multi_{count}",
                "images": images,
                "input_tokens": 20 + count * 5,
                "expected_output_tokens": count * 20,
                "expected_answer": ", ".join(shapes_in_prompt)
            })
    
    # Save vision prompts
    with open(images_dir / "vision_prompts.json", "w") as f:
        json.dump(vision_prompts, f, indent=2)
    
    print(f"Generated {len(vision_prompts)} vision prompts with images")
    print(f"Created {len(os.listdir(images_dir)) - 1} image files")  # -1 for the JSON file
    
    return vision_prompts

if __name__ == "__main__":
    generate_vision_prompts()