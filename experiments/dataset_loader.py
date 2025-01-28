
import pandas as pd

from sklearn.model_selection import train_test_split

from datasets import RSNADataset, WSIDataset

CLUSTER_DATA_DIR = '/data/datasets/'
HOME_DATA_DIR = '/home/fran/data/datasets/'


def load_dataset(config, mode='train', bag_size_limit=None):
    name = config.dataset_name
    val_prop = config.val_prop
    n_samples = None
    seed = config.seed

    dataset_id = name.split('-')[0]

    if dataset_id == "rsna":
        
        if 'features' in name:
            # rsna-features_<model_name>
            features_dir_name = name.split('-')[1]
            data_path = f'{HOME_DATA_DIR}/RSNA_ICH/features/{features_dir_name}/'
            processed_data_path = f'{HOME_DATA_DIR}/RSNA_ICH/MIL_processed/{features_dir_name}/'
        else:
            data_path = f'{HOME_DATA_DIR}/RSNA_ICH/original/'
            processed_data_path = f'{HOME_DATA_DIR}/RSNA_ICH/MIL_processed/original/'
        
        if mode=='train_val':
            csv_path = f'{CLUSTER_DATA_DIR}/RSNA_ICH/bags_train.csv'
        else:
            csv_path = f'{CLUSTER_DATA_DIR}/RSNA_ICH/bags_test.csv'
        
        dataset = RSNADataset(data_path=data_path, processed_data_path=processed_data_path, csv_path=csv_path, n_samples=n_samples, use_slice_distances=config.use_inst_distances, adj_mat_mode=config.adj_mat_mode)

        if mode=='train_val':        
            bags_labels = dataset.get_bag_labels()
            len_ds = len(bags_labels)
            
            idx = list(range(len_ds))
            idx_train, idx_val = train_test_split(idx, test_size=val_prop, random_state=seed, stratify=bags_labels)

            train_dataset = dataset.subset(idx_train)
            val_dataset = dataset.subset(idx_val)
        elif mode=='test':
            test_dataset = dataset
    elif dataset_id == "panda":
        
        directory = 'PANDA/PANDA_original/'
        patches_dir_name = name.split('-')[1]
        features_dir_name = name.split('-')[2]
        main_data_path = f'{HOME_DATA_DIR}/{directory}/{patches_dir_name}/'

        def read_csv_fn(csv_path, partition='train'):
            df = pd.read_csv(csv_path)
            df = df[df['Partition']==partition]
            bag_names_list = df['image_id'].values
            bag_labels_list = df['isup_grade'].astype(int).apply(lambda x : 0 if x == 0 else 1).values
            return bag_names_list, bag_labels_list
        
        csv_path = f'{CLUSTER_DATA_DIR}/{directory}/original/wsi_labels.csv'
        partition = 'train' if mode=='train_val' else 'test'
        
        dataset = WSIDataset(
            main_data_path, 
            csv_path, 
            features_dir_name, 
            use_patch_distances=config.use_inst_distances, 
            adj_mat_mode=config.adj_mat_mode, 
            bag_size_limit=bag_size_limit,
            read_csv_fn= lambda csv_path: read_csv_fn(csv_path, partition=partition)
        )

        if mode=='train_val':

            bags_labels = dataset.get_bag_labels()
            len_ds = len(bags_labels)

            idx = list(range(len_ds))
            idx_train, idx_val = train_test_split(idx, test_size=val_prop, random_state=seed, stratify=bags_labels)

            train_dataset = dataset.subset(idx_train)
            val_dataset = dataset.subset(idx_val)
        
        elif mode=='test':
            test_dataset = dataset

    elif dataset_id == "camelyon16":

        directory = 'CAMELYON16'
        patches_dir_name = name.split('-')[1]
        features_dir_name = name.split('-')[2]
        main_data_path = f'{HOME_DATA_DIR}/{directory}/{patches_dir_name}/'       

        if mode=='train_val':
            csv_path = f'{CLUSTER_DATA_DIR}/{directory}/original/train.csv'
        elif mode=='test':
            csv_path = f'{CLUSTER_DATA_DIR}/{directory}/original/test.csv'
        
        def read_csv_fn(csv_path):
            df = pd.read_csv(csv_path)
            bag_names_list = df['wsi_name'].apply(lambda x: x.split('.')[0]).values
            bag_labels_list = df['wsi_label'].astype(int).values
            return bag_names_list, bag_labels_list
        
        dataset = WSIDataset(
            main_data_path, 
            csv_path, 
            features_dir_name, 
            use_patch_distances=config.use_inst_distances, 
            adj_mat_mode=config.adj_mat_mode, 
            bag_size_limit=bag_size_limit,
            read_csv_fn=read_csv_fn
        )

        if mode=='train_val':

            bags_labels = dataset.get_bag_labels()
            len_ds = len(bags_labels)

            idx = list(range(len_ds))
            idx_train, idx_val = train_test_split(idx, test_size=val_prop, random_state=seed, stratify=bags_labels)

            train_dataset = dataset.subset(idx_train)
            val_dataset = dataset.subset(idx_val)
        
        elif mode=='test':
            test_dataset = dataset
    else:
        raise ValueError(f"Dataset {name} not supported")
    
    if mode=='train_val':
        return train_dataset, val_dataset
    elif mode=='test':
        return test_dataset

# def load_train_val_dataset(config):
#     name = config.dataset_name
#     val_prop = config.val_prop
#     n_samples = None
#     seed = config.seed
#     if "rsna" in name:
        
#         if 'features' in name:
#             # rsna-features_<model_name>
#             features_dir_name = name.split('-')[1]
#             data_path = f'{HOME_DATA_DIR}/RSNA_ICH/features/{features_dir_name}/'
#             processed_data_path = f'{HOME_DATA_DIR}/RSNA_ICH/MIL_processed/{features_dir_name}/'
#         else:
#             data_path = f'{HOME_DATA_DIR}/RSNA_ICH/original/'
#             processed_data_path = f'{HOME_DATA_DIR}/RSNA_ICH/MIL_processed/original/'
        
#         csv_path = f'{CLUSTER_DATA_DIR}/RSNA_ICH/bags_train.csv'
#         dataset = RSNADataset(data_path=data_path, processed_data_path=processed_data_path, csv_path=csv_path, n_samples=n_samples, use_slice_distances=config.use_inst_distances, adj_mat_mode=config.adj_mat_mode)
        
#         bags_labels = dataset.get_bag_labels()
#         len_ds = len(bags_labels)
        
#         idx = list(range(len_ds))
#         idx_train, idx_val = train_test_split(idx, test_size=val_prop, random_state=seed, stratify=bags_labels)

#         train_dataset = dataset.subset(idx_train)
#         val_dataset = dataset.subset(idx_val)

#     elif "panda" in name:

#         if 'features' in name:
#             patches_dir_name = name.split('-')[1]
#             features_dir_name = name.split('-')[2]
#             main_data_path = f'{HOME_DATA_DIR}/PANDA/PANDA_original/{patches_dir_name}/'
#             csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_original/original/train.csv'
#         else:
#             raise ValueError(f"PANDA dataset only supports features")

#         dataset = WSIDataset(main_data_path, csv_path, features_dir_name, use_patch_distances=config.use_inst_distances, adj_mat_mode=config.adj_mat_mode)
#         bags_labels = dataset.get_bag_labels()
#         len_ds = len(bags_labels)

#         idx = list(range(len_ds))
#         idx_train, idx_val = train_test_split(idx, test_size=val_prop, random_state=seed, stratify=bags_labels)

#         train_dataset = dataset.subset(idx_train)
#         val_dataset = dataset.subset(idx_val)
        
#         # panda-<patch_dir>-<features_dir>
#         # Ex: panda-patches_512-features_resnet18

#         # patch_dir = name.split('-')[1]

#         # if 'features' in name:
#         #     features_dir_name = name.split('-')[2]
#         #     data_path = f'{HOME_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/features/{features_dir_name}/'
#         #     processed_data_path = f'{HOME_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/MIL_processed/{features_dir_name}/'
#         #     csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/train_val_patches.csv'
#         #     aux_csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/wsi_labels.csv'
#         #     # train_csv_path = f'{HOME_DATA_DIR}/PANDA/PANDA_downsample/patches_512/train_patches.csv'
#         #     # val_csv_path = f'{HOME_DATA_DIR}/PANDA/PANDA_downsample/patches_512/val_patches.csv'
#         # else:
#         #     data_path = f'{HOME_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/images/'
#         #     processed_data_path = f'{HOME_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/MIL_processed/images/'
#         #     csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/train_val_patches.csv'
#         #     aux_csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/wsi_labels.csv'
#         #     # train_csv_path = '{HOME_DATA_DIR}/PANDA/PANDA_downsample/patches_512/train_patches.csv'
#         #     # val_csv_path = '{HOME_DATA_DIR}/PANDA/PANDA_downsample/patches_512/val_patches.csv'
    
#         # dataset = PandaDataset(data_path=data_path, processed_data_path=processed_data_path, csv_path=csv_path, aux_csv_path=aux_csv_path, n_samples=n_samples, use_patch_distances=config.use_inst_distances, adj_mat_mode=config.adj_mat_mode)
        
#         # bags_labels = dataset.get_bag_labels()
#         # len_ds = len(bags_labels)

#         # idx = list(range(len_ds))
#         # idx_train, idx_val = train_test_split(idx, test_size=val_prop, random_state=seed, stratify=bags_labels)

#         # train_dataset = dataset.subset(idx_train)
#         # val_dataset = dataset.subset(idx_val)

#     elif 'camelyon16' in name:

#         # camelyon16-<patch_dir>-<features_dir>
#         # Ex: camelyon16-patches_512_preset-features_resnet50_bt
        
#         if 'features' in name:
#             patches_dir_name = name.split('-')[1]
#             features_dir_name = name.split('-')[2]
#             main_data_path = f'{HOME_DATA_DIR}/CAMELYON16/{patches_dir_name}/'
#             csv_path = f'{CLUSTER_DATA_DIR}/CAMELYON16/original/train.csv'
#         else:
#             raise ValueError(f"CAMELYON16 dataset only supports features")

#         dataset = WSIDataset(main_data_path, csv_path, features_dir_name, use_patch_distances=config.use_inst_distances, adj_mat_mode=config.adj_mat_mode)
#         bags_labels = dataset.get_bag_labels()
#         len_ds = len(bags_labels)

#         idx = list(range(len_ds))
#         idx_train, idx_val = train_test_split(idx, test_size=val_prop, random_state=seed, stratify=bags_labels)

#         train_dataset = dataset.subset(idx_train)
#         val_dataset = dataset.subset(idx_val)
    
#     elif name=="mnist":
#         train_dataset = MNISTMILDataset(subset='train', bag_size=10)
#         val_dataset = MNISTMILDataset(subset='train', bag_size=10)
#     elif name=='mnist_correlated':
#         train_dataset = CorrelatedToyMILDataset(name='mnist', subset='train', num_bags=1500, obj_labels=[i for i in range(2,6)], bag_size=10, seed=10*seed)
#         val_dataset = CorrelatedToyMILDataset(name='mnist', subset='train', num_bags=200, obj_labels=[i for i in range(2,6)], bag_size=10 , seed=10*seed+1)
#     elif name=="cifar100_correlated":
#         # num_bags = 4000
#         num_bags = 8000
#         train_dataset = CorrelatedToyMILDataset(name='cifar100', subset='train', num_bags=num_bags, obj_labels=[i for i in range(20, 30)], bag_size=30, seed=10*seed)
#         val_dataset = CorrelatedToyMILDataset(name='cifar100', subset='train', num_bags=int(0.2*num_bags), obj_labels=[i for i in range(20, 30)], bag_size=30, seed=10*seed+1)
#     elif name=='cifar100_correlated_v2':
#         num_bags = 4000
#         train_dataset = CorrelatedToyMILDataset(name='cifar100', subset='train', num_bags=num_bags, obj_labels=[i for i in range(20, 30)], bag_size=30, use_inst_labels=True)
#         val_dataset = CorrelatedToyMILDataset(name='cifar100', subset='train', num_bags=int(0.2*num_bags), obj_labels=[i for i in range(20, 30)], bag_size=30, use_inst_labels=True)
#     else:
#         raise ValueError(f"Dataset {name} not supported")
    
#     return train_dataset, val_dataset

# def load_test_dataset(config, bag_size_limit=None):
#     name = config.dataset_name
#     n_samples = None
#     seed = config.seed
#     if 'rsna' in name:
#         if 'features' in name:
#             # rsna-features_<model_name>
#             features_dir_name = name.split('-')[1]
#             data_path = f'{HOME_DATA_DIR}/RSNA_ICH/features/{features_dir_name}/'
#             processed_data_path = f'{HOME_DATA_DIR}/RSNA_ICH/MIL_processed/{features_dir_name}/'
#         else:
#             data_path = f'{HOME_DATA_DIR}/RSNA_ICH/original/'
#             processed_data_path = f'{HOME_DATA_DIR}/RSNA_ICH/MIL_processed/original/'
#         csv_path = f'{CLUSTER_DATA_DIR}/RSNA_ICH/bags_test.csv'
#         test_dataset = RSNADataset(data_path=data_path, processed_data_path=processed_data_path, csv_path=csv_path, n_samples=n_samples, use_slice_distances=config.use_inst_distances, adj_mat_mode=config.adj_mat_mode)  
#     elif 'panda' in name:

#         if 'features' in name:
#             patches_dir_name = name.split('-')[1]
#             features_dir_name = name.split('-')[2]
#             main_data_path = f'{HOME_DATA_DIR}/PANDA/PANDA_original/{patches_dir_name}/'
#             csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_original/original/test.csv'
#         else:
#             raise ValueError(f"PANDA dataset only supports features")
#         test_dataset = WSIDataset(main_data_path, csv_path, features_dir_name, use_patch_distances=config.use_inst_distances, adj_mat_mode=config.adj_mat_mode, bag_size_limit=bag_size_limit)

#         # patch_dir = name.split('-')[1]

#         # if 'features' in name:
#         #     features_dir_name = name.split('-')[2]
#         #     data_path = f'{HOME_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/features/{features_dir_name}/'
#         #     processed_data_path = f'{HOME_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/MIL_processed/{features_dir_name}/'
#         #     csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/test_patches.csv'
#         #     aux_csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/wsi_labels.csv'
#         # else:
#         #     data_path = f'{HOME_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/images/'
#         #     processed_data_path = f'{HOME_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/MIL_processed/images/'
#         #     csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/test_patches.csv'
#         #     aux_csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/wsi_labels.csv'

#         # test_dataset = PandaDataset(data_path=data_path, processed_data_path=processed_data_path, csv_path=csv_path, aux_csv_path=aux_csv_path, n_samples=n_samples, use_patch_distances=config.use_inst_distances, adj_mat_mode=config.adj_mat_mode)  
#     elif "camelyon16" in name:
        
#         # camelyon16-<patch_dir>-<features_dir>
#         # Ex: camelyon16-patches_512_preset-features_resnet50_bt
        
#         if 'features' in name:
#             patches_dir_name = name.split('-')[1]
#             features_dir_name = name.split('-')[2]
#             main_data_path = f'{HOME_DATA_DIR}/CAMELYON16/{patches_dir_name}/'
#             csv_path = f'{CLUSTER_DATA_DIR}/CAMELYON16/original/test.csv'
#         else:
#             raise ValueError(f"CAMELYON16 dataset only supports features")
#         test_dataset = CamelyonDataset(main_data_path, csv_path, features_dir_name, use_patch_distances=config.use_inst_distances, adj_mat_mode=config.adj_mat_mode, bag_size_limit=bag_size_limit)
#     elif name=="mnist":
#         test_dataset = MNISTMILDataset(subset='test', bag_size=10)
#     elif name=='mnist_correlated':
#         test_dataset = CorrelatedToyMILDataset(name='mnist', subset='test', num_bags=200, obj_labels=[i for i in range(2, 6)], bag_size=10, seed=10*seed+2)
#     elif name=="cifar100_correlated":
#         test_dataset = CorrelatedToyMILDataset(name='cifar100', subset='test', num_bags=800, obj_labels=[i for i in range(20, 30)], bag_size=30, seed=10*seed+2)
#     elif name=='cifar100_correlated_v2':
#         test_dataset = CorrelatedToyMILDataset(name='cifar100', subset='test', num_bags=800, obj_labels=[i for i in range(20, 30)], bag_size=30, use_inst_labels=True)
#     else:
#         raise ValueError(f"Dataset {name} not supported")
#     return test_dataset
