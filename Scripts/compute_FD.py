import argparse

from Helpers.compute_RETFD import calculate_fid_from_paths
from Helpers.compute_clean_fid import compute_fid
from Helpers.utils import load_yaml

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that takes a YAML config.")
    parser.add_argument('--yaml', type=str, required=True, help='Path to the YAML config')
    parser.add_argument('--bs', type=int, required=True, help='batch size for the RET-FD computation')

    args = parser.parse_args()

    print(f"The YAML config path is: {args.yaml}")

    config = load_yaml(args.yaml)

    ret_FID = lambda a,b: calculate_fid_from_paths(a ,b , batch_size= args.bs, backbone="RETFound")
    fid = compute_fid

    model_pred_path = config['generation_path_save']
    root_dataset_dir = config.get('root_dir','../../Databases')

    datasets = config.get('test_dataset',["DRTiD"])

    model_pred_paths = [model_pred_path + f"_[012]*/{dataset}/*.*" for dataset in datasets]


    folder_to_evaluate = model_pred_paths

    paths_folders_1 = [f"{root_dataset_dir}{dataset}/images/*.*" for dataset in datasets]

    fid_scores = fid(paths_folders_1, folder_to_evaluate)
    print(f"clean fid | {folder_to_evaluate} | {fid_scores}" )

    fid_scores = ret_FID(paths_folders_1, folder_to_evaluate)
    print(f"RETFiD | {folder_to_evaluate} | {fid_scores}")


