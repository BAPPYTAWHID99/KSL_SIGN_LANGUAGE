import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from generate_gloss_map import scan_metadata


class GlossMapMetadataTests(unittest.TestCase):
    def test_scans_label_json_and_zip_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            label_directory = root / "[라벨]"
            label_directory.mkdir()
            (label_directory / "labels.json").write_text(
                json.dumps({"WORD0001": {"word": "안녕하세요"}}, ensure_ascii=False),
                encoding="utf-8",
            )
            with zipfile.ZipFile(label_directory / "labels.zip", "w") as archive:
                archive.writestr(
                    "WORD0002.csv",
                    "gloss\n감사합니다\n",
                )

            glosses, failures = scan_metadata(root)

        self.assertEqual(failures, [])
        self.assertEqual(
            glosses,
            {"WORD0001": "안녕하세요", "WORD0002": "감사합니다"},
        )


if __name__ == "__main__":
    unittest.main()
