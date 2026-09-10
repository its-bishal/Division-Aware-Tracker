import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import motmetrics as mm
import pandas as pd

def parse_xml_gt(xml_path):
    """
    Parses the ground truth XML file and returns a list of bounding boxes per frame.
    Format of return: dict mapping frame_idx (1-based) to list of [track_id, x, y, w, h]
    """
    gt_data = {}
    
    if not os.path.exists(xml_path):
        print(f"Error: Ground truth file {xml_path} not found.")
        return gt_data
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    images = root.find('images')
    if images is None:
        return gt_data
        
    for image in images.findall('image'):
        src = image.find('src').text
        # src is like "1-50\image1.png". We extract the number from "image1.png"
        filename = os.path.basename(src.replace('\\', '/'))
        # try:
        #     frame_idx = int(filename.replace('image', '').replace('.png', ''))
        # except ValueError:
        #     continue
        match = re.search(r'(\d+)(?=\.[^.]+$)', filename)
        if not match:
            continue
        frame_idx = int(match.group(1))
            
        gt_data[frame_idx] = []
        
        bbs = image.find('boundingboxes')
        if bbs is None:
            continue
            
        for bb in bbs.findall('boundingbox'):
            x = float(bb.find('x_left_top').text)
            y = float(bb.find('y_left_top').text)
            w = float(bb.find('width').text)
            h = float(bb.find('height').text)
            
            class_name = bb.find('class_name')
            if class_name is not None:
                track_elem = class_name.find('track_id')
                if track_elem is not None:
                    track_id = int(track_elem.text)
                    gt_data[frame_idx].append([track_id, x, y, w, h])
                    
    return gt_data

def load_tracker_csv(csv_path):
    """
    Loads tracker output in MOTChallenge format.
    Format of return: dict mapping frame_idx (1-based) to list of [track_id, x, y, w, h]
    """
    pred_data = {}
    if not os.path.exists(csv_path):
        print(f"Error: Tracker output file {csv_path} not found.")
        return pred_data
        
    df = pd.read_csv(csv_path, header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x_3d', 'y_3d', 'z_3d'])
    for _, row in df.iterrows():
        frame_idx = int(row['frame'])
        if frame_idx not in pred_data:
            pred_data[frame_idx] = []
        pred_data[frame_idx].append([int(row['id']), row['x'], row['y'], row['w'], row['h']])
        
    return pred_data

def evaluate(gt_data, pred_data):
    """
    Computes MOT metrics using motmetrics.
    """
    acc = mm.MOTAccumulator(auto_id=True)
    
    all_frames = sorted(list(set(list(gt_data.keys()) + list(pred_data.keys()))))
    
    for frame in all_frames:
        gt_frame = gt_data.get(frame, [])
        pred_frame = pred_data.get(frame, [])
        
        gt_ids = [item[0] for item in gt_frame]
        gt_boxes = [[item[1], item[2], item[3], item[4]] for item in gt_frame]
        
        pred_ids = [item[0] for item in pred_frame]
        pred_boxes = [[item[1], item[2], item[3], item[4]] for item in pred_frame]
        
        # Calculate distance matrix (IoU)
        dist_matrix = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        
        acc.update(gt_ids, pred_ids, dist_matrix)
        
    mh = mm.metrics.create()
    
    summary = mh.compute(acc, metrics=['mota', 'motp', 'num_switches', 'idp', 'idr', 'idf1', 'mostly_tracked', 'mostly_lost'], name='CellTracker')
    
    # Format summary
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap={
            'mota': 'MOTA',
            'motp': 'MOTP',
            'num_switches': 'ID_Switches',
            'idp': 'ID_Precision',
            'idr': 'ID_Recall',
            'idf1': 'IDF1',
            'mostly_tracked': 'MT',
            'mostly_lost': 'ML'
        }
    )
    
    print("\n--- Tracking Evaluation Results ---")
    print(strsummary)
    print("-----------------------------------")

if __name__ == "__main__":
    # gt_path = "/home/bishal/Downloads/datasets/RBCdataset/cellsFourSlits/fourSlits_annotated_cells_and_tracks_frames_1_150.xml"
    gt_path = '/home/bishal/Downloads/datasets/RBCdataset/cellsUshapedChannel/uShape_annotated_cells_and_tracks_frames_1_300.xml'
    pred_path = "tracks.csv"
    
    print("Parsing ground truth...")
    gt_data = parse_xml_gt(gt_path)
    print(f"Found {len(gt_data)} frames in ground truth.")
    
    print("Parsing predictions...")
    pred_data = load_tracker_csv(pred_path)
    print(f"Found {len(pred_data)} frames in predictions.")
    
    if len(gt_data) > 0 and len(pred_data) > 0:
        print("Computing metrics...")
        evaluate(gt_data, pred_data)
    else:
        print("Evaluation aborted due to missing data.")
