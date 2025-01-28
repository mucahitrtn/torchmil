# !/bin/bash
# nohup ./upload_rsna.sh > output/upload_rsna.log 2>&1 &

DATA_PATH=/home/fran/data/datasets/RSNA_ICH

# rm $DATA_PATH/*.tar.gz
# rm $DATA_PATH/MIL_processed/*.tar.gz

huggingface-cli upload --repo-type dataset RSNA_ICH_MIL $DATA_PATH/splits.csv ./dataset/splits.csv

file_list=(labels slice_labels)

for file in ${file_list[@]}; do

    # if $DATA_PATH/$file.tar.gz not exists, create it
    if [ ! -f $DATA_PATH/$file.tar.gz ]; then
        tar --exclude='*.h5' -czvf $DATA_PATH/$file.tar.gz -C $DATA_PATH/$file ./
    fi

    huggingface-cli upload --repo-type dataset RSNA_ICH_MIL $DATA_PATH/$file.tar.gz ./dataset/$file.tar.gz
done

features_list=(features_resnet50 features_resnet18 features_vit_b_32)

for feature in ${features_list[@]}; do

    if [ -d $DATA_PATH/MIL_processed/$feature ]; then

        # if $DATA_PATH/MIL_processed/$feature.tar.gz not exists, create it
        if [ ! -f $DATA_PATH/MIL_processed/$feature.tar.gz ]; then
            tar --exclude='*.h5' -czvf $DATA_PATH/MIL_processed/$feature.tar.gz -C $DATA_PATH/MIL_processed/$feature ./
        fi

        huggingface-cli upload --repo-type dataset RSNA_ICH_MIL $DATA_PATH/MIL_processed/$feature.tar.gz ./dataset/features/$feature.tar.gz
    fi
done

echo "Done!"