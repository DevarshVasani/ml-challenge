
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        use_fast=False,
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("ByT5 tokenizer unexpectedly has no EOS token")

    manifest_path = Path(args.pair_manifest)
    pair_dir = Path(args.pair_dir)
    shards = _manifest_shards(manifest_path, pair_dir)

    lengths = array("I")
    total_pairs = 0
    truncated_pairs = 0
    cut_locations: Counter[str] = Counter()
    address_a_affected = 0
    address_b_affected = 0
    country_totals: Counter[str] = Counter()
    country_truncated: Counter[str] = Counter()

    for shard in tqdm(shards, desc="ByT5 truncation shards"):
        frame = pd.read_parquet(shard)
        total_pairs += len(frame)
        if "country_a" in frame.columns:
            for value, count in frame["country_a"].fillna("").astype(str).value_counts(dropna=False).items():
                country_totals[str(value)] += int(count)

        for rows in _iter_batches(frame, args.batch_size):
            texts = [serialize_pair_fields(row) for row in rows]
            encoded = tokenizer(
                texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]

            for row, ids in zip(rows, encoded):
                token_length = len(ids)
                lengths.append(token_length)
                if token_length <= args.max_length:
                    continue

                truncated_pairs += 1
                country_truncated[str(row.get("country_a", "") or "")] += 1
                # ByT5 maps raw UTF-8 content bytes directly; one position is kept for EOS.
                location, a_affected, b_affected = _utf8_cut_location(row, args.max_length - 1)
                cut_locations[location] += 1
                address_a_affected += int(a_affected)
                address_b_affected += int(b_affected)

    if total_pairs != len(lengths):
        raise RuntimeError(f"internal row-count mismatch: read {total_pairs}, measured {len(lengths)}")

    length_values = np.frombuffer(lengths, dtype=np.uint32)
    if len(length_values):
        percentiles = np.percentile(length_values, [50, 90, 95, 99])
        length_stats = {
            "p50": float(percentiles[0]),
            "p90": float(percentiles[1]),
            "p95": float(percentiles[2]),
            "p99": float(percentiles[3]),
            "max": int(length_values.max()),
        }
    else:
        length_stats = {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}

    by_country = {}
    for country in sorted(country_totals):
        total = int(country_totals[country])
        cut = int(country_truncated[country])
        by_country[country] = {
            "pairs": total,
            "truncated_pairs": cut,
            "truncation_rate": cut / total if total else 0.0,
        }

    report = {
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "max_length": args.max_length,
        "pair_manifest": str(manifest_path),
        "pair_dir": str(pair_dir),
        "total_pairs": total_pairs,
        "truncated_pairs": truncated_pairs,
        "truncation_rate": truncated_pairs / total_pairs if total_pairs else 0.0,
        "token_length_stats": length_stats,
        "cut_locations": dict(sorted(cut_locations.items())),
        "address_a_affected_pairs": address_a_affected,
        "address_b_affected_pairs": address_b_affected,
        "address_a_affected_rate": address_a_affected / total_pairs if total_pairs else 0.0,
        "address_b_affected_rate": address_b_affected / total_pairs if total_pairs else 0.0,
        "by_country": by_country,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
