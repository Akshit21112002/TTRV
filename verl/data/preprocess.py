import os

import datasets

import os
import datasets


def make_map_fn(split, source=None):
    def process_fn(example, idx):
        if source is None:
            data_source = example.pop("source")
        else:
            data_source = source

        # Construct question
        question = example.pop("prompt")
        image_path = example.pop("image_path", None)  # Allow missing image_path
        solution = example.pop("answer")

        # Only include images if image_path is present and file exists
        images = []
        if image_path and os.path.exists(image_path):
            images = [{"image": image_path}]
        else:
            images = None  

        data = {
            "data_source": "GPQA-TTT",
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": f"{data_source}-{idx}",
            },
            "images": images,  
        }

        return data

    return process_fn


if __name__ == '__main__':

    data_source = "dtd_20"  # put the dataset folder name here


    train_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, 'train.json'), split='train')
    test_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, 'test.json'), split='train')


    train_dataset = train_dataset.map(function=make_map_fn("train", data_source), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", data_source), with_indices=True)

    train_dataset.to_parquet(os.path.join(data_source, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(data_source, 'test.parquet'))