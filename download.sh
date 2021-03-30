mkdir -p data/vqa2
cd data/vqa2
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

unzip "*mscoco.zip"
rm *mscoco.zip

mkdir -p ../data/coco
cd ../data/coco
wget https://webia.lip6.fr/~dancette/shortcut-detection/image_to_detection.json
