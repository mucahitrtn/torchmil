# !/bin/bash
# nohup ./upload_panda.sh > output/upload_panda.log 2>&1 &

DATA_PATH=/home/fran/data/datasets/PANDA/PANDA_original/patches_512_preset/

rm $DATA_PATH/*.tar.gz
rm $DATA_PATH/features/*.tar.gz

huggingface-cli upload --repo-type dataset PANDA_MIL $DATA_PATH/splits.csv ./dataset/splits.csv

file_list=(labels coords patch_labels)

for file in ${file_list[@]}; do

    # if $DATA_PATH/$file.tar.gz not exists, create it
    if [ ! -f $DATA_PATH/$file.tar.gz ]; then
        tar --exclude='*.h5' -czvf $DATA_PATH/$file.tar.gz -C $DATA_PATH/$file ./
    fi

    huggingface-cli upload --repo-type dataset PANDA_MIL $DATA_PATH/$file.tar.gz ./dataset/patches_512/$file.tar.gz
done

features_list=(features_UNI features_resnet50_bt features_resnet50)

for feature in ${features_list[@]}; do

    # if $DATA_PATH/features/$feature.tar.gz not exists, create it
    if [ ! -f $DATA_PATH/features/$feature.tar.gz ]; then
        tar --exclude='*.h5' -czvf $DATA_PATH/features/$feature.tar.gz -C $DATA_PATH/features/$feature ./
    fi

    huggingface-cli upload --repo-type dataset PANDA_MIL $DATA_PATH/features/$feature.tar.gz ./dataset/patches_512/features/$feature.tar.gz
done

echo "Done!"
