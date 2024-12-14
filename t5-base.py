import os
import argparse
import time
import functools

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Block
from datasets import Dataset

def train_function(dataset_url, model_name, num_samples, input_length, output_length, batch_size, num_epochs, learning_rate):
    parameters = {
        "DATASET_URL": dataset_url,
        "MODEL_NAME": model_name,
    }
    # [1] Setup PyTorch distributed and get the distributed parameters.
    dist.init_process_group("nccl")
    pod_name = os.environ.get("POD_NAME", "unknown-pod")
    # Pod 이름 패턴 기반 RANK 계산
    if "master" in pod_name:
        rank = 0  # Master Pod의 Rank는 항상 0
    elif "worker" in pod_name:
        rank = int(pod_name.split("-")[-1]) + 1  # Worker Rank는 1부터 시작
    else:
        raise ValueError(f"Unknown pod name format: {pod_name}")

    local_rank = rank % torch.cuda.device_count()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Local rank identifies the GPU number inside the pod.
    torch.cuda.set_device(local_rank)

    print(f"FSDP Training for WORLD_SIZE: {world_size}, RANK: {rank}, LOCAL_RANK: {local_rank}, LEARNING_RATE: {learning_rate}")

    # [2] Prepare the Wikihow dataset
    dataset = Dataset.from_csv(dataset_url)
    dataset = dataset.select(list(range(0, num_samples)))

    class wikihow(torch.utils.data.Dataset):
        def __init__(
            self,
            dataset,
            tokenizer,
            num_samples,
            input_length,
            output_length,
        ):
            self.dataset = dataset
            self.input_length = input_length
            self.tokenizer = tokenizer
            self.output_length = output_length

        def __len__(self):
            return self.dataset.shape[0]

        def clean_text(self, text):
            # Dataset contains empty values.
            if text is None:
                return ""
            text = text.replace("Example of text:", "")
            text = text.replace("Example of Summary:", "")
            text = text.replace("\n", "")
            text = text.replace("``", "")
            text = text.replace('"', "")

            return text

        def convert_to_features(self, example_batch):
            # Tokenize text and headline (as pairs of inputs).
            input_ = self.clean_text(example_batch["text"]) if "text" in example_batch else ""
            target_ = self.clean_text(example_batch["headline"]) if "headline" in example_batch else ""

            source = self.tokenizer.batch_encode_plus(
                [input_],
                max_length=self.input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            targets = self.tokenizer.batch_encode_plus(
                [target_],
                max_length=self.output_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return source, targets

        def __getitem__(self, index):
            source, targets = self.convert_to_features(self.dataset[index])

            source_ids = source["input_ids"].squeeze()
            target_ids = targets["input_ids"].squeeze()

            src_mask = source["attention_mask"].squeeze()
            target_mask = targets["attention_mask"].squeeze()

            return {
                "source_ids": source_ids,
                "source_mask": src_mask,
                "target_ids": target_ids,
                "target_mask": target_mask,
            }

    # [3] Get the T5 pre-trained model and tokenizer.
    # Since this script is run by multiple workers, we should print results only for the worker with RANK=0.
    if rank == 0:
        print(f"Downloading the {parameters['MODEL_NAME']} model")

    MODEL_CACHE_DIR = '/data/model_cache'  # 모델 캐싱 디렉토리
    DATASET_CACHE_DIR = '/data/wikihow'  # 데이터셋 캐싱 디렉토리
    dataset_path = os.path.normpath(os.path.join(DATASET_CACHE_DIR, "wikihowAll.csv"))

    model = T5ForConditionalGeneration.from_pretrained(parameters["MODEL_NAME"], cache_dir=MODEL_CACHE_DIR)
    tokenizer = T5Tokenizer.from_pretrained(parameters["MODEL_NAME"], cache_dir=MODEL_CACHE_DIR)
    model.to(local_rank)
    # [4] Download the Wikihow dataset.
    if rank == 0:
        print("Downloading the Wikihow dataset")

    if not os.path.exists(dataset_path):
        if rank == 0:
            print(f"Downloading dataset to {dataset_path}")
        os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
        dataset = Dataset.from_csv(dataset_url)
        dataset.to_pandas().to_csv(dataset_path, index=False)
    else:
        dataset = Dataset.from_csv(dataset_path)

    dataset = wikihow(dataset, tokenizer, num_samples, input_length, output_length)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset),
    )

    # [5] Setup model with FSDP.
    # Model is on CPU before input to FSDP.
    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Block,
        },
    )
    model = FSDP(
        model,
        auto_wrap_policy=t5_auto_wrap_policy,
        device_id=torch.cuda.current_device(),
    )

    # [6] Start training.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    t0 = time.time()
    if rank == 0:
        print("Training is started...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        fsdp_loss = torch.zeros(2).to(local_rank)

        for batch in train_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)

            optimizer.zero_grad()

            output = model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=batch["target_ids"],
            )
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            fsdp_loss[0] += loss.item()
            fsdp_loss[1] += len(batch)

        dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        train_accuracy = fsdp_loss[0] / fsdp_loss[1]

        if rank == 0:
            print(f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}")
        scheduler.step()

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print(f"FSDP training time: {int(time.time() - t0)} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_url", type=str, default="https://public-nlp-datasets.s3.us-west-2.amazonaws.com/wikihowAll.csv", help="URL for the dataset to be used for training")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Name of the pre-trained T5 model to be used")
    parser.add_argument("--num_samples", type=int, default=1500, help="Total number of samples to be used from the dataset")
    parser.add_argument("--input_length", type=int, default=512, help="Maximum length of the input sequences")
    parser.add_argument("--output_length", type=int, default=150, help="Maximum length of the output sequences")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of samples per batch during training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=0.002, help="Learning rate for the optimizer")
    args = parser.parse_args()

    train_function(
        dataset_url=args.dataset_url,  # URL for the dataset to be used for training
        model_name=args.model_name,  # Name of the pre-trained T5 model to be used
        num_samples=args.num_samples,  # Total number of samples to be used from the dataset
        input_length=args.input_length,  # Maximum length of the input sequences
        output_length=args.output_length,  # Maximum length of the output sequences
        batch_size=args.batch_size,  # Number of samples per batch during training
        num_epochs=args.num_epochs,  # Number of epochs to train the model
        learning_rate=args.learning_rate,  # Learning rate for the optimizer
    )
