import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import Seq2SeqTrainingArguments, HfArgumentParser, AutoConfig, AutoTokenizer
from transformers.trainer_utils import is_main_process, set_seed

from models import T5ForConditionalGeneration, ABSAPrefixForConditionalGeneration
from trainer import MultiTaskTrainer
from utils.data_utils_mix import ABSADataset
# from utils.data_utils import ABSADataset
from utils.eval_utils import parse_and_score

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="../../PretrainModel/t5-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cls_model_name_or_path: str = field(
        default="../../PretrainModel/t5-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: str = field(
        default='unified', metadata={"help": "acos/asqp/aste/tasd/unified "}
    )
    dataset: str = field(
        default='laptop14', metadata={"help": "laptop14/rest14/rest15/rest16 "}
    )
    train_path: Optional[str] = field(
        default=None, metadata={"help": "The file path of train data"}
    )
    valid_path: Optional[str] = field(
        default=None, metadata={"help": "The file path of valid data"},
    )
    test_path: Optional[str] = field(
        default=None, metadata={"help": "The file path of test data"},
    )
    max_length: int = field(
        default=300, metadata={"help": "The max padding length of source text and target text"}
    )
    num_beams: int = field(
        default=1, metadata={"help": "greedy search"}
    )
    shot_ratio_index: Optional[str] = field(
        default="-1[+]-1[+]1", metadata={"help": "1[+]-1[+]1->1-shot"}
    )
    lowercase: Optional[bool] = field(
        default=None, metadata={"help": "lowercase sentences"}
    )
    sort_label: Optional[bool] = field(
        default=None, metadata={"help": "sort tuple by order of appearance"}
    )
    ctrl_token: str = field(
        default='post', metadata={"help": "combine sentence and orders"})
    topk: int = field(
        default=1, metadata={"help": "order number"}
    )
    combined: Optional[bool] = field(
        default=True, metadata={"help": "combined sentences with explicit sentiment and implicit sentiment"}
    )
    virtual_token: Optional[bool] = field(
        default=True, metadata={"help": "use virtual tokens"}
    )
    implicit_token: Optional[bool] = field(
        default=True, metadata={"help": "use implicit tokens"}
    )

    def __post_init__(self):
        if self.dataset is None and self.train_path is None and self.valid_path is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        shot, ratio, index = self.shot_ratio_index.split("[+]")
        shot, ratio, index = int(shot), float(ratio), int(index)
        assert shot in [-1, 1, 5, 10] and ratio in [-1, 0.01, 0.05, 0.1]
        print(ratio)
        # self.full_supervise = True if shot == -1 and ratio == -1 else False
        self.full_supervise = True
        # name_mapping = {"laptop14": "14lap", "rest14": "14res", "rest15": "15res", "rest16": "16res"}
        if shot != -1:
            self.train_path = f'./data_my/{self.task_name}/{self.dataset}-shot/{shot}/seed3407/train.txt'
            self.valid_path = f'./data_my/{self.task_name}/{self.dataset}-shot/{shot}/seed3407/dev.txt'
            self.test_path = f'./data_my/{self.task_name}/{self.dataset}-shot/{shot}/seed3407/test.txt'
        elif ratio != -1:
            self.train_path = f'./data/ratio/{self.task_name}/{self.dataset}-ratio/{ratio}/seed3407/train.txt'
            self.valid_path = f'./data/ratio/{self.task_name}/{self.dataset}-ratio/{ratio}/seed3407/dev.txt'
            self.test_path = f'./data/ratio/{self.task_name}/{self.dataset}-ratio/{ratio}/seed3407/test.txt'
        else:
            # self.train_path = f'./data/{self.task_name}/{self.dataset}/train.txt'
            # self.valid_path = f'./data/{self.task_name}/{self.dataset}/dev.txt'
            # self.test_path = f'./data/{self.task_name}/{self.dataset}/test.txt'
            self.train_path = f'./data/{self.task_name}/{self.dataset}/train.txt'
            self.valid_path = f'./data/{self.task_name}/{self.dataset}/dev.txt'
            self.test_path = f'./data/{self.task_name}/{self.dataset}/test.txt'

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )


@dataclass
class ABSASeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Arguments for our model in training procedure
    """
    constraint_decoding: bool = field(default=False, metadata={"help": "Whether to Constraint Decoding or not."})
    alpha: float = field(default=0.5, metadata={"help": "adjust the loss weight of ao_template and oa_template"})


def main():
    # 参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ABSASeq2SeqTrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script, and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     # print("sys.argv[1]: ",sys.argv[1])
    #     model_args, data_args, training_args = parser.parse_json_file(
    #         json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args = parser.parse_json_file(json_file="./args.json")
    # print(f"model_args: {model_args}")
    # print(f"data_args: {data_args}")
    # print(f"training_args: {training_args}")
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # torch.manual_seed(training_args.seed)
    config_t5 = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # pass args
    config_t5.max_length = data_args.max_length
    config_t5.num_beams = data_args.num_beams

    # 模型
    tokenizer_t5 = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    pretrained_model_t5 = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config_t5,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # implicit_virtual_tokens = ['<implicit_vtoken_a>', '<implicit_vtoken_a1>', '<implicit_vtoken_c>', '<implicit_vtoken_s>', '<implicit_vtoken_o>', '<implicit_vtoken_o1>', '<implicit_sentence>']
    # explicit_virtual_tokens = ['<explicit_vtoken_a>', '<explicit_vtoken_a1>', '<explicit_vtoken_c>', '<explicit_vtoken_s>', '<explicit_vtoken_o>', '<explicit_vtoken_o1>', '<explicit_sentence>']
    implicit_virtual_tokens = ['<implicit_vtoken_a>', '<implicit_vtoken_c>', '<implicit_vtoken_s>',
                               '<implicit_vtoken_o>']
    explicit_virtual_tokens = ['<explicit_vtoken_a>', '<explicit_vtoken_c>', '<explicit_vtoken_s>',
                               '<explicit_vtoken_o>']
    tokenizer_t5.add_tokens(implicit_virtual_tokens + explicit_virtual_tokens)
    # 为T5模型扩展词表并初始化这些虚拟token的嵌入
    pretrained_model_t5.resize_token_embeddings(len(tokenizer_t5))

    model = ABSAPrefixForConditionalGeneration(
        opt=model_args,
        t5_model=pretrained_model_t5,
        t5_encoder=pretrained_model_t5.encoder
    )

    if not training_args.do_train:
        logger.info("load checkpoint of IsABSAModel !")
        # model.load_state_dict(torch.load(f"{training_args.output_dir}/checkpoint-6832/pytorch_model.bin"))
        model.load_state_dict(torch.load(""))

    train_dataset = ABSADataset(tokenizer_t5=tokenizer_t5,
                                data_path=data_args.train_path,
                                args=data_args,
                                data_type="train",
                                )
    eval_dataset = ABSADataset(tokenizer_t5=tokenizer_t5,
                               data_path=data_args.valid_path,
                               args=data_args,
                               data_type="eval",
                               )
    test_dataset = ABSADataset(tokenizer_t5=tokenizer_t5,
                               data_path=data_args.test_path,
                               args=data_args,
                               data_type="test",
                               )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # if labels[0][-1] == -100:
        #     labels = [sublist[:-1] for sublist in labels]
        # if labels[0][-1] == -100:
        #     labels = labels[:, :399]
        preds = tokenizer_t5.batch_decode(preds, skip_special_tokens=True)
        # print(preds[:5])
        # print(f"pred: {preds}")
        # pdb.set_trace()
        labels = np.where(labels != -100, labels, tokenizer_t5.pad_token_id)
        labels = tokenizer_t5.batch_decode(labels, skip_special_tokens=True)
        # valid_labels = [label for label in labels if all(0 <= id < tokenizer_t5.vocab_size for id in label)]
        # labels = valid_labels
        # result = parse_and_score(preds, labels, data_args.data_format)
        result = parse_and_score(preds, labels, data_args)
        return result

    # 训练
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # optimizers=(optimizer, scheduler),
        tokenizer=tokenizer_t5,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None
    )

    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        output_train_file = os.path.join(
            training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(
                training_args.output_dir, "trainer_state.json"))

    if training_args.do_predict:
        data_args.max_length = 1024
        logger.info(f"*** Test constraint decoding: {training_args.constraint_decoding}***")
        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.max_length,
            num_beams=data_args.num_beams,
            constraint_decoding=training_args.constraint_decoding,
        )
        # with open("./slgm-rest14-46.txt", 'w', encoding='utf-8') as f:
        #     for idx, prediction in enumerate(test_results.predictions):
        #         decoded_text = tokenizer.decode(prediction, skip_special_tokens=True)
        #         # print(f"Example {idx + 1} Decoded Text: {decoded_text}")
        #         f.write(decoded_text + '\n')
        test_metrics = test_results.metrics
        test_metrics["test_loss"] = round(test_metrics["test_loss"], 4)
        output_test_result_file = os.path.join(training_args.output_dir, "test_results_seq2seq.txt")
        if trainer.is_world_process_zero():
            with open(output_test_result_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in sorted(test_metrics.items()):
                    logger.info(f"{key} = {value}")
                    writer.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
