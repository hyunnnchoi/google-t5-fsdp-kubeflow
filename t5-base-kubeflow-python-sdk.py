# pip install uvicorn git+https://github.com/kubeflow/training-operator.git#subdirectory=sdk/python

from kubeflow.training import TrainingClient

def train_function(parameters):
    import os
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

    # [1] Setup PyTorch distributed and get the distributed parameters.
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Local rank identifies the GPU number inside the pod.
    torch.cuda.set_device(local_rank)

    print(
        f"FSDP Training for WORLD_SIZE: {world_size}, RANK: {rank}, LOCAL_RANK: {local_rank}"
    )

    # [2] Prepare the Wikihow dataset
    class wikihow(torch.utils.data.Dataset):
        def __init__(
            self,
            tokenizer,
            num_samples,
            input_length,
            output_length,
        ):

            self.dataset = Dataset.from_csv(parameters["DATASET_URL"])
            self.dataset = self.dataset.select(list(range(0, num_samples)))
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
            input_ = self.clean_text(example_batch["text"])
            target_ = self.clean_text(example_batch["headline"])

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

    model = T5ForConditionalGeneration.from_pretrained(parameters["MODEL_NAME"])
    tokenizer = T5Tokenizer.from_pretrained(parameters["MODEL_NAME"])

    # [4] Download the Wikihow dataset.
    if rank == 0:
        print("Downloading the Wikihow dataset")

    dataset = wikihow(tokenizer, 1500, 512, 150)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    t0 = time.time()
    if rank == 0:
        print("Training is started...")

    for epoch in range(1, 3):
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

    if rank == 0:
        print(f"FSDP training time: {int(time.time() - t0)} seconds")

job_name = "fsdp-fine-tuning"

parameters = {
    "DATASET_URL": "https://public-nlp-datasets.s3.us-west-2.amazonaws.com/wikihowAll.csv",
    "MODEL_NAME": "t5-base",
}
# Create the PyTorchJob.
TrainingClient().create_job(
    name=job_name,
    train_func=train_function,
    parameters=parameters,
    num_workers=4,  # 단일 노드에서 4개의 Worker
    num_procs_per_worker=1,  # 각 Worker당 1개의 GPU 사용
    resources_per_worker={"gpu": 1},  # Worker당 1개의 GPU 할당
    packages_to_install=[
        "transformers==4.38.2",
        "datasets==2.21.0",
        "SentencePiece==0.2.0",
    ],
)
