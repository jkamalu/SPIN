import os
import json
import argparse
import tempfile
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jrdb-coco", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    
    args = parser.parse_args()
    
    path_to_img = os.path.join("/cvgl/u/jkamalu/jrdb_coco", args.jrdb_coco, args.scene, "images")
    path_to_ann = os.path.join("/cvgl/u/jkamalu/jrdb_coco", args.jrdb_coco, args.scene, "annotations.json")
    
    with open(path_to_ann, "r") as reader:
        ann = json.load(reader)
        
    fnames = {annotation["id"]:annotation["file_name"] for annotation in ann["images"]}
    
    bboxes = defaultdict(list)
    for annotation in ann["annotations"]:
        bboxes[fnames[annotation["image_id"]]].append(annotation["bbox"])

    with tempfile.NamedTemporaryFile("wt+") as writer:
        writer.write(json.dumps(bboxes))
        writer.seek(0)
    
        os.chdir("/cvgl/u/jkamalu/SPIN")
            
        os.system(f"mkdir -p {os.path.join('/cvgl/u/jkamalu/SPIN/examples', args.jrdb_coco, args.scene)}")

        os.system(f"python3 /cvgl/u/jkamalu/SPIN/demo_scene.py \
                    --checkpoint=data/model_checkpoint.pt \
                    --imgdir={path_to_img}\
                    --detections={writer.name} \
                    --logdir={os.path.join('/cvgl/u/jkamalu/SPIN/examples', args.jrdb_coco, args.scene)}")
