            if len(ids) <= max_length:
                clipped.append(ids)
                continue
            shortened = ids[:max_length]
            if eos_token_id is not None:
                shortened[-1] = int(eos_token_id)
            clipped.append(shortened)

        padded = self.tokenizer.pad(
            {"input_ids": clipped},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": padded["input_ids"].to(self.device, non_blocking=True),
            "attention_mask": padded["attention_mask"].to(self.device, non_blocking=True),
            "truncated": int(truncated),
        }

    def logits(self, batch: Mapping[str, Any]) -> Any:
        self._require_initialized()
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    def parameters(self):
        self._require_initialized()
        return self.model.parameters()

    def to_device(self, device: str) -> "ByT5EncoderPairAdapter":
        import torch

        self._require_initialized()
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def supports_gradient_checkpointing(self) -> bool:
        return True

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        super().set_gradient_checkpointing(enabled)
        self._require_initialized()
        if enabled:
            self.model.encoder.gradient_checkpointing_enable()
        else:
            self.model.encoder.gradient_checkpointing_disable()

    def save_pretrained(self, output_dir: str) -> None:
        import torch
        import transformers

        self._require_initialized()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        config_payload = self.config_dict()
        (output / "adapter_config.json").write_text(
            json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        metadata = {
            "architecture": _ADAPTER_ARCHITECTURE,
            "original_checkpoint": self.config.checkpoint,
            "requested_revision": self.config.revision,
            "resolved_revision": _resolved_revision(self.model.encoder, self.tokenizer),
            "max_length": int(self.config.max_length),
            "d_model": int(self.model.encoder.config.d_model),
            "output_size": 2,
            "pooling": "masked_mean",
            "serialization_separator": PAIR_SERIALIZATION_SEPARATOR,
            "encoder_class": type(self.model.encoder).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
        (output / "adapter_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        self.tokenizer.save_pretrained(str(output / "tokenizer"))
        self.model.encoder.save_pretrained(str(output / "encoder"))

        classifier_state = {
            key: value.detach().cpu()
            for key, value in self.model.classifier.state_dict().items()
        }
        temporary = output / ".classifier.pt.tmp"
        torch.save(classifier_state, temporary)
        os.replace(temporary, output / "classifier.pt")

    @classmethod
    def load_pretrained(cls, output_dir: str, *, map_location: str = "cpu") -> "ByT5EncoderPairAdapter":
        import torch
        from transformers import AutoTokenizer, T5EncoderModel

        output = Path(output_dir)
        required = [
            output / "adapter_config.json",
            output / "classifier.pt",
            output / "tokenizer",
            output / "encoder",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"incomplete ByT5 adapter checkpoint; missing: {missing}")

        config = AdapterConfig(**json.loads((output / "adapter_config.json").read_text(encoding="utf-8")))
        instance = cls(config)
        instance.tokenizer = AutoTokenizer.from_pretrained(
            str(output / "tokenizer"),
            use_fast=False,
            local_files_only=True,
        )
        encoder = T5EncoderModel.from_pretrained(
            str(output / "encoder"),
            local_files_only=True,
        )
        instance.model = _build_pair_classifier(encoder)

        # Load on CPU first for predictable peak memory, then move the completed model.
        classifier_state = torch.load(output / "classifier.pt", map_location="cpu", weights_only=True)
        instance.model.classifier.load_state_dict(classifier_state)
        instance.to_device(map_location)