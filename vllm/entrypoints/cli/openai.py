# SPDX-License-Identifier: Apache-2.0
# Commands that act as an interactive OpenAI API client

import argparse
import os
import signal
import sys
from typing import List, Optional, Tuple

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.lora.request import LoRARequest
from vllm.utils import FlexibleArgumentParser


def _register_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def _interactive_cli(args: argparse.Namespace) -> Tuple[str, OpenAI]:
    _register_signal_handlers()

    base_url = args.url
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    openai_client = OpenAI(api_key=api_key, base_url=base_url)

    if args.model_name:
        model_name = args.model_name
    else:
        available_models = openai_client.models.list()
        model_name = available_models.data[0].id

    print(f"Using model: {model_name}")

    return model_name, openai_client


def chat(system_prompt: Optional[str], model_name: str,
         client: OpenAI) -> None:
    conversation: List[ChatCompletionMessageParam] = []
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})

    print("Please enter a message for the chat model:")
    while True:
        try:
            input_message = input("> ")
        except EOFError:
            return
        conversation.append({"role": "user", "content": input_message})

        chat_completion = client.chat.completions.create(model=model_name,
                                                         messages=conversation)

        response_message = chat_completion.choices[0].message
        output = response_message.content

        conversation.append(response_message)  # type: ignore
        print(output)


def _add_query_options(
        parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/v1",
        help="url of the running OpenAI-Compatible RESTful API server")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=("The model name used in prompt completion, default to "
              "the first model in list models API call."))
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for OpenAI services. If provided, this api key "
            "will overwrite the api key obtained through environment variables."
        ))
    return parser


class ChatCommand(CLISubcommand):
    """The `chat` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "chat"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        model_name, client = _interactive_cli(args)
        system_prompt = args.system_prompt
        conversation: List[ChatCompletionMessageParam] = []
        if system_prompt is not None:
            conversation.append({"role": "system", "content": system_prompt})

        print("Please enter a message for the chat model:")
        while True:
            try:
                input_message = input("> ")
            except EOFError:
                return
            conversation.append({"role": "user", "content": input_message})

            chat_completion = client.chat.completions.create(
                model=model_name, messages=conversation)

            response_message = chat_completion.choices[0].message
            output = response_message.content

            conversation.append(response_message)  # type: ignore
            print(output)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        chat_parser = subparsers.add_parser(
            "chat",
            help="Generate chat completions via the running API server",
            usage="vllm chat [options]")
        _add_query_options(chat_parser)
        chat_parser.add_argument(
            "--system-prompt",
            type=str,
            default=None,
            help=("The system prompt to be added to the chat template, "
                  "used for models that support system prompts."))
        return chat_parser


class CompleteCommand(CLISubcommand):
    """The `complete` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "complete"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        model_name, client = _interactive_cli(args)
        print("Please enter prompt to complete:")
        while True:
            input_prompt = input("> ")
            completion = client.completions.create(model=model_name,
                                                   prompt=input_prompt)
            output = completion.choices[0].text
            print(output)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        complete_parser = subparsers.add_parser(
            "complete",
            help=("Generate text completions based on the given prompt "
                  "via the running API server"),
            usage="vllm complete [options]")
        _add_query_options(complete_parser)
        return complete_parser

class BenchmarkThroughputCommand(CLISubcommand):
    def __init___(self):
        self.name = "benchmark-throughput"
        super().__init__()

    def cmd(args: argparse.Namespace) -> None:
        model_name, client = _interactive_cli(args)
        print(args)
        import random
        random.seed(args.seed)

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("HuggingFace transformers library is not installed. "
                              "Build vLLM properly from source: "
                              "https://docs.vllm.ai/en/latest/getting_started/installation/index.html")
        # Sample the requests.
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code)
        if args.dataset is None:
            vocab_size = tokenizer.vocab_size
            requests = []

            for _ in range(args.num_prompts):

                request_tokenizer = tokenizer
                lora_request: Optional[LoRARequest] = None
                if args.enable_lora:
                    lora_request, lora_tokenizer = get_random_lora_request(args)
                    if lora_tokenizer:
                        request_tokenizer = lora_tokenizer

                # Synthesize a prompt with the given input length.
                candidate_ids = [
                    random.randint(0, vocab_size - 1)
                    for _ in range(args.input_len)
                ]
                # As tokenizer may add additional tokens like BOS, we need to try
                # different lengths to get the desired input length.
                for _ in range(5):  # Max attempts to correct
                    candidate_prompt = request_tokenizer.decode(candidate_ids)
                    tokenized_len = len(request_tokenizer.encode(candidate_prompt))

                    if tokenized_len == args.input_len:
                        break

                    # Adjust length based on difference
                    diff = args.input_len - tokenized_len
                    if diff > 0:
                        candidate_ids.extend([
                            random.randint(100, vocab_size - 100)
                            for _ in range(diff)
                        ])
                    else:
                        candidate_ids = candidate_ids[:diff]
                requests.append(
                    SampleRequest(prompt=candidate_prompt,
                                  prompt_len=args.input_len,
                                  expected_output_len=args.output_len,
                                  lora_request=lora_request))
        else:
            requests = sample_requests(tokenizer, args)

        is_multi_modal = any(request.multi_modal_data is not None
                             for request in requests)
        if args.backend == "vllm":
            if args.async_engine:
                elapsed_time = uvloop.run(
                    run_vllm_async(
                        requests,
                        args.n,
                        AsyncEngineArgs.from_cli_args(args),
                        args.disable_frontend_multiprocessing,
                    ))
            else:
                elapsed_time = run_vllm(requests, args.n,
                                        EngineArgs.from_cli_args(args))
        elif args.backend == "hf":
            assert args.tensor_parallel_size == 1
            elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                                  args.hf_max_batch_size, args.trust_remote_code)
        elif args.backend == "mii":
            elapsed_time = run_mii(requests, args.model, args.tensor_parallel_size,
                                   args.output_len)
        else:
            raise ValueError(f"Unknown backend: {args.backend}")
        total_num_tokens = sum(request.prompt_len + request.expected_output_len
                               for request in requests)
        total_output_tokens = sum(request.expected_output_len
                                  for request in requests)
        if is_multi_modal:
            print("\033[91mWARNING\033[0m: Multi-modal request detected. The "
                  "following metrics are not accurate because image tokens are not"
                  " counted. See vllm-project/vllm/issues/9778 for details.")
            # TODO(vllm-project/vllm/issues/9778): Count molti-modal token length.
        print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
              f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
              f"{total_output_tokens / elapsed_time:.2f} output tokens/s")

        # Output JSON results if specified
        if args.output_json:
            results = {
                "elapsed_time": elapsed_time,
                "num_requests": len(requests),
                "total_num_tokens": total_num_tokens,
                "requests_per_second": len(requests) / elapsed_time,
                "tokens_per_second": total_num_tokens / elapsed_time,
            }
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=4)
            save_to_pytorch_benchmark_format(args, results)


def cmd_init() -> List[CLISubcommand]:
    return [ChatCommand(), CompleteCommand()]
