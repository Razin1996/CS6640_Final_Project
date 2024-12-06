import os
import cv2
import torch
import csv
import numpy as np
from PIL import Image
from datetime import datetime
from skimage.io import imsave
from skimage.measure import label, regionprops
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# Load the segmentation model
def load_model(model_config):
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_config)
    model = SegformerForSemanticSegmentation.from_pretrained(model_config)
    model.eval()
    model.cuda()
    return model, feature_extractor

# Perform inference and return largest segmented mask of target class
def inference(image, model, feature_extractor, target_class=21):
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].cuda()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    predictions = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()
    class_mask = (predictions == target_class).astype(np.uint8)
    labeled_mask = label(class_mask)
    regions = regionprops(labeled_mask)
    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        largest_mask = (labeled_mask == largest_region.label).astype(np.uint8)
    else:
        largest_mask = np.zeros_like(class_mask)
    return largest_mask

# Overlay segmented mask on the original image
def overlay_mask_on_image(image, mask, opacity=100):
    opacity = max(0, min(255, opacity))
    mask = mask.astype(np.uint8)
    resized_mask = Image.fromarray(mask).resize(image.size, resample=Image.NEAREST)
    rgba_mask = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
    rgba_mask[..., :3] = np.array([0, 0, 255])
    rgba_mask[..., 3] = (np.array(resized_mask) * opacity)
    rgba_image = np.array(image.convert("RGBA"))
    combined_image = Image.alpha_composite(Image.fromarray(rgba_image), Image.fromarray(rgba_mask))
    return combined_image

# Extract timestamp from filename
def extract_timestamp(filename):
    timestamp_str = '_'.join(filename.split('_')[2:4])  # Adjust indices if necessary
    try:
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
        return timestamp
    except ValueError as e:
        #print(f"Error processing filename {filename}: {e}")
        return None

# Draw regions of interest (ROIs) and count water pixels
def draw_rois_and_get_water_pixels(image_path, roi_coords, lower_blue=np.array([110,50,50]), upper_blue=np.array([130,255,255])):
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
    water_pixels_counts = []
    image_with_rois = image.copy()
    for roi in roi_coords:
        cv2.rectangle(image_with_rois, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 4)
        roi_mask = np.zeros_like(mask)
        roi_mask[roi[1]:roi[3], roi[0]:roi[2]] = 255
        segmented_roi = cv2.bitwise_and(mask, mask, mask=roi_mask)
        water_pixels_counts.append(np.sum(segmented_roi == 255))
    return water_pixels_counts, image_with_rois

# Main function to process images, segment, and analyze ROIs
def main(folder_path, model_config, output_dir, csv_file_path, rois, target_class=21):
    model, feature_extractor = load_model(model_config)
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'LeftBank', 'RightBank_1', 'RightBank_2'])
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path).convert("RGB")
                pred_mask = inference(image, model, feature_extractor, target_class)
                combined_image = overlay_mask_on_image(image, pred_mask)
                overlay_image_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_overlay.png')
                combined_image.save(overlay_image_path)
                water_pixels, image_with_rois = draw_rois_and_get_water_pixels(overlay_image_path, rois)
                timestamp = extract_timestamp(filename)
                writer.writerow([timestamp] + water_pixels)
                cv2.imwrite(overlay_image_path, np.array(image_with_rois))
                print(f"Processed {filename}")
    print(f"Data saved to {csv_file_path}")

if __name__ == "__main__":
    folder_path = "image"
    model_config = "nvidia/segformer-b5-finetuned-ade-640-640"
    output_dir = "segmented_image"
    csv_file_path = "water_pixels.csv"
    rois = [(750, 250, 1200, 700), (2000, 300, 2450, 750)]
    target_class = 21
    main(folder_path, model_config, output_dir, csv_file_path, rois, target_class)
