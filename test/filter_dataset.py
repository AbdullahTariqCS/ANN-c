import os
from PIL import Image
import shutil

PATH_TO_DATASET = '..\\archive\\3\\train'
PATH_TO_NEW_DATASET = '..\\dataset' 

def convert_jpg_to_pgm(jpg_path, pgm_path=None):
    with Image.open(jpg_path) as img:
        gray_img = img.convert("L")
        
        if pgm_path is None:
            base = os.path.splitext(jpg_path)[0]
            pgm_path = base + ".pgm"
        
        gray_img.save(pgm_path, format="PPM")
        print(f"Saved PGM file to: {pgm_path}")
    
if __name__ == '__main__':
    for dir in list(range(48, 58)) + list(range(65, 91)): 
        dir_path = os.path.join(PATH_TO_NEW_DATASET, str(dir))
        if os.path.exists(dir_path): 
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)        


        for root, _, files in os.walk(os.path.join(PATH_TO_DATASET, str(dir))): 
            for file in files:
                print(file)
                src_path = os.path.join(root, file)
                filename_wo_ext = os.path.splitext(file)[0]
                dst_path = os.path.join(dir_path, filename_wo_ext + '.pgm')
                convert_jpg_to_pgm(src_path, dst_path)
