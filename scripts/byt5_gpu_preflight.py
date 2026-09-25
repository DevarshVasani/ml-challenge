    parser.add_argument("--output", default="artifacts/byt5-preflight.json")
    args = parser.parse_args()

    import torch
    import transformers

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    training = config.get("training", config)
    result = {
        "config": args.config,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "device_count": torch.cuda.device_count(),
        "model_check_requested": args.load_model,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "total_vram_gb": props.total_memory / (1024 ** 3),
            "capability": list(torch.cuda.get_device_capability(0)),
        }

    disk = shutil.disk_usage(Path.cwd())
    result["disk_free_gb"] = disk.free / (1024 ** 3)

    artifacts = {
        "train": _pair_manifest_check(Path("artifacts/neural-data/train_pairs_manifest.json"), "train_pairs"),
        "threshold": _pair_manifest_check(Path("artifacts/neural-data/threshold_pairs_manifest.json"), "threshold_pairs"),
        "final": _pair_manifest_check(Path("artifacts/neural-data/final_pairs_manifest.json"), "final_pairs"),
        "query_manifest": {"path": "artifacts/neural-data/query_manifest.json", "ok": Path("artifacts/neural-data/query_manifest.json").is_file()},
        "candidate_manifest": {"path": "artifacts/neural-candidates/manifest.json", "ok": Path("artifacts/neural-candidates/manifest.json").is_file()},
    }
    result["artifacts"] = artifacts

    hard_failures = []
    if not result["cuda_available"]:
        hard_failures.append("CUDA is not available")
    if str(training.get("precision", "")).lower() == "bf16" and not result["bf16_supported"]:
        hard_failures.append("BF16 was requested but this GPU/driver does not report BF16 support")
    if not artifacts["train"]["ok"]:
        hard_failures.append("training pair manifest/shards are incomplete")
    if not artifacts["query_manifest"]["ok"]:
        hard_failures.append("query manifest is missing")
    if not args.allow_training_only:
        for key in ("threshold", "final", "candidate_manifest"):
            if not artifacts[key]["ok"]:
                hard_failures.append(f"required shared artifact is incomplete: {key}")

    if args.load_model and not hard_failures:
        from src.neural_adapters import AdapterConfig, get_adapter_class
        from src.neural_contracts import seed_everything

        adapter_type = str(training.get("adapter_type", config.get("adapter_type", "byt5")))
        adapter_config = AdapterConfig(
            adapter_type=adapter_type,
            checkpoint=str(training.get("checkpoint", config.get("checkpoint", "google/byt5-small"))),
            revision=training.get("revision", config.get("revision")),
            max_length=int(training.get("max_length", config.get("max_length", 512))),
            device=str(training.get("device", "cuda")),
        )
        seed_everything(int(training.get("seed", 42)))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        adapter = get_adapter_class(adapter_type).from_config(adapter_config, execute=True)
        adapter.to_device(str(training.get("device", "cuda")))
        adapter.model.eval()
        rows = [
            {
                "name_a": "Café Shree & Sons Pvt Ltd",
                "address_a": "12 MG Road, Ahmedabad 380001",
                "country_a": "India",
                "name_b": "Cafe Shree and Sons Private Limited",
                "address_b": "12 M.G. Rd Ahmedabad 380001",
                "country_b": "India",
            },
            {
                "name_a": "München Handel GmbH",
                "address_a": "Hauptstraße 7",
                "country_a": "Germany",
                "name_b": "Munchen Handel GMBH",
                "address_b": "Hauptstrasse 7",
                "country_b": "Germany",
            },
        ]
        batch = adapter.collate(rows)
        dtype = torch.bfloat16 if str(training.get("precision", "")).lower() == "bf16" else torch.float32
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
            logits = adapter.logits(batch)
        if tuple(logits.shape) != (2, 2) or not torch.isfinite(logits).all():
            hard_failures.append(f"ByT5 forward sanity failed: shape={tuple(logits.shape)}")
        result["model_check"] = {
            "logits_shape": list(logits.shape),
            "truncated": int(batch["truncated"]),
            "peak_allocated_gb": torch.cuda.max_memory_allocated(0) / (1024 ** 3),
            "peak_reserved_gb": torch.cuda.max_memory_reserved(0) / (1024 ** 3),
        }

    result["ok"] = not hard_failures
    result["failures"] = hard_failures
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if hard_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
