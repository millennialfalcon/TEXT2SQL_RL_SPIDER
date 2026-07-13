"""CPU-only smoke test for terminal, file, and JSONL diagnostics."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from debug_output import JSONLWriter, build_logger


def main() -> None:
    """Write representative diagnostics and verify their persisted contents."""
    with TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        jsonl_path = output_dir / "completions.jsonl"
        log_path = output_dir / "training.log"

        writer = JSONLWriter(jsonl_path)
        logger = build_logger(log_path)

        records = [
            {
                "sample_id": 0,
                "raw_completion": "Here is the answer: SELECT 1;",
                "extracted_sql": "Here is the answer: SELECT 1;",
                "reward": 0.0,
                "outcome": "execution_error",
                "error": 'near "Here": syntax error',
            },
            {
                "sample_id": 1,
                "raw_completion": "SELECT 1;",
                "extracted_sql": "SELECT 1;",
                "reward": 0.2,
                "outcome": "wrong_result",
                "error": None,
            },
        ]

        for record in records:
            writer.write(record)
            logger.info(
                "sample_id=%s reward=%s outcome=%s error=%s",
                record["sample_id"],
                record["reward"],
                record["outcome"],
                record["error"],
            )

        loaded_records = [
            json.loads(line) for line in jsonl_path.read_text().splitlines()
        ]
        assert loaded_records == records
        assert "execution_error" in log_path.read_text()
        assert "wrong_result" in log_path.read_text()

    print("logging and JSONL diagnostics ok")


if __name__ == "__main__":
    main()
