from huggingface_hub import hf_hub_download, snapshot_download
# hf_hub_download(repo_id="KevinMathew/stereo4d-lefteye-perspective", filename="fleurs.py", repo_type="dataset")
snapshot_download(repo_id="KevinMathew/stereo4d-lefteye-perspective", repo_type="dataset", local_dir="/mnt/shared-storage-user/idc2-shared/binyanrui/vggt_datasets6/stereo4d/huggingface")

