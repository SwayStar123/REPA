import argparse
import json
import os
import xml.etree.ElementTree as ET
from typing import List
import multiprocessing as mp
from functools import partial

import tqdm


def collect_png_files(root_dir: str) -> List[str]:
    """Return list of .png image paths (relative to root_dir) sorted alphabetically."""
    files = []
    for current_root, _dirs, fnames in os.walk(root_dir):
        for fname in fnames:
            if fname.lower().endswith('.png'):
                rel_path = os.path.relpath(os.path.join(current_root, fname), start=root_dir)
                files.append(rel_path.replace('\\', '/'))
    return sorted(files)


def parse_xml_worker(xml_path: str) -> tuple[str, str]:
    """Worker function to parse a single XML file and extract synset."""
    try:
        tree = ET.parse(xml_path)
        elem = tree.find('.//object/name')
        if elem is None or elem.text is None:
            return os.path.basename(xml_path), None
        return os.path.basename(xml_path), elem.text.strip()
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return os.path.basename(xml_path), None


def build_synset_mapping(xml_dir: str, workers: int = 32) -> dict[str, int]:
    """Build synset -> class_idx mapping using alphabetical ordering of synset ids in xml_dir."""
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    xml_paths = [os.path.join(xml_dir, f) for f in xml_files]
    
    synsets = set()
    print(f"Parsing {len(xml_files)} XML files with {workers} workers...")
    
    with mp.Pool(workers) as pool:
        results = pool.map(parse_xml_worker, xml_paths)
    
    for xml_fname, synset in results:
        if synset is not None:
            synsets.add(synset)
    
    sorted_synsets = sorted(synsets)
    return {synset: idx for idx, synset in enumerate(sorted_synsets)}


def process_image_worker(args: tuple) -> tuple[int, str, int]:
    """Worker function to process a single image and extract its label."""
    idx, rel_png, xml_dir, synset_to_class = args
    
    try:
        # Get sorted XML files (this is consistent across all workers)
        xml_files_sorted = sorted(os.listdir(xml_dir))
        if idx >= len(xml_files_sorted):
            raise IndexError(f'Image index {idx} exceeds XML files count {len(xml_files_sorted)}')
        
        xml_path = os.path.join(xml_dir, xml_files_sorted[idx])
        tree = ET.parse(xml_path)
        elem = tree.find('.//object/name')
        
        if elem is None or elem.text is None:
            raise RuntimeError(f'Missing synset in {xml_files_sorted[idx]}')
        
        synset = elem.text.strip()
        
        if synset not in synset_to_class:
            raise RuntimeError(f'Unknown synset {synset} in {xml_files_sorted[idx]}')
        
        label = synset_to_class[synset]
        return idx, rel_png, label
    
    except Exception as e:
        print(f"Error processing image {idx} ({rel_png}): {e}")
        return idx, rel_png, None


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate dataset.json labels for processed ImageNet validation set")
    parser.add_argument('--images-dir', required=True, help='Path to processed validation images directory (root that contains 00000/, 00001/, ...)')
    parser.add_argument('--xml-dir', required=True, help='Path to ILSVRC/Annotations/CLS-LOC/val directory with XML files')
    parser.add_argument('--output', default='dataset.json', help='Output JSON filename (relative to images-dir)')
    parser.add_argument('--workers', type=int, default=32, help='Number of parallel workers for processing')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing images')
    return parser.parse_args()


def main():
    args = parse_args()

    images_dir = args.images_dir
    xml_dir = args.xml_dir
    output_fname = os.path.join(images_dir, args.output)
    workers = args.workers
    batch_size = args.batch_size

    print('Collecting PNG files...')
    png_files = collect_png_files(images_dir)
    print(f'Found {len(png_files)} images')

    print('Building synset mapping...')
    synset_to_class = build_synset_mapping(xml_dir, workers)
    print(f'Detected {len(synset_to_class)} unique synsets')

    print(f'Processing images with {workers} workers in batches of {batch_size}...')
    labels_list = [None] * len(png_files)
    
    # Process images in batches to avoid memory issues
    for batch_start in tqdm.tqdm(range(0, len(png_files), batch_size), desc='Processing batches'):
        batch_end = min(batch_start + batch_size, len(png_files))
        batch_args = [
            (idx, png_files[idx], xml_dir, synset_to_class) 
            for idx in range(batch_start, batch_end)
        ]
        
        with mp.Pool(workers) as pool:
            batch_results = pool.map(process_image_worker, batch_args)
        
        # Store results
        for idx, rel_png, label in batch_results:
            if label is None:
                raise RuntimeError(f'Failed to process image {idx} ({rel_png})')
            labels_list[idx] = [rel_png, label]

    assert all(entry is not None for entry in labels_list), "Some labels were not processed"

    with open(output_fname, 'w') as f:
        json.dump({'labels': labels_list}, f)
    print(f'Saved labels to {output_fname}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # Required for robust multiprocessing
    main() 