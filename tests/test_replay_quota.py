import json
import runpy
import subprocess
from unittest.mock import patch

import pytest

from tools.download_replays import (
    ReplayDatasetEntry,
    append_manifest_entry,
    bytes_to_gb,
    download_replays,
    parse_kaggle_files_table,
    would_exceed_cap,
)


def test_bytes_to_gb_uses_binary_gib():
    assert bytes_to_gb(1024 ** 3) == 1.0


def test_would_exceed_cap_blocks_over_100_gb():
    assert would_exceed_cap(current_bytes=99 * 1024 ** 3, next_bytes=2 * 1024 ** 3, cap_gb=100)
    assert not would_exceed_cap(current_bytes=90 * 1024 ** 3, next_bytes=5 * 1024 ** 3, cap_gb=100)


def test_would_exceed_cap_boundary_byte_over_cap():
    assert would_exceed_cap(current_bytes=100 * 1024 ** 3, next_bytes=1, cap_gb=100)


def test_append_manifest_entry_creates_new_manifest(tmp_path):
    manifest_path = tmp_path / "data" / "manifest.json"
    entry = ReplayDatasetEntry(
        source="kaggle",
        local_path="replays/episode_1.json",
        size_bytes=1024,
        discussion_id="d1",
    )

    append_manifest_entry(manifest_path, entry, cap_gb=100)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data == [
        {
            "source": "kaggle",
            "local_path": "replays/episode_1.json",
            "size_bytes": 1024,
            "discussion_id": "d1",
        }
    ]


def test_append_manifest_entry_preserves_existing_entries(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([
            {
                "source": "kaggle",
                "local_path": "replays/episode_1.json",
                "size_bytes": 10,
                "discussion_id": "d1",
            }
        ]),
        encoding="utf-8",
    )
    entry = ReplayDatasetEntry("kaggle", "replays/episode_2.json", 20, "d2")

    append_manifest_entry(manifest_path, entry)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[1]["discussion_id"] == "d2"


def test_append_manifest_entry_rejects_over_cap(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    existing = {
        "source": "kaggle",
        "local_path": "replays/episode_1.json",
        "size_bytes": 100 * 1024 ** 3,
        "discussion_id": "d1",
    }
    manifest_path.write_text(json.dumps([existing]), encoding="utf-8")

    with pytest.raises(ValueError, match="cap"):
        append_manifest_entry(
            manifest_path,
            ReplayDatasetEntry("kaggle", "replays/episode_2.json", 1, "d2"),
            cap_gb=100,
        )


def test_append_manifest_entry_rejects_invalid_manifest_shape(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"no": "list"}), encoding="utf-8")

    with pytest.raises(ValueError, match="list"):
        append_manifest_entry(
            manifest_path,
            ReplayDatasetEntry("kaggle", "replays/episode_2.json", 1, "d2"),
        )


def test_append_manifest_entry_rejects_missing_or_invalid_size_bytes(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([
            {"source": "kaggle", "local_path": "x", "discussion_id": "d1"},
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="size_bytes"):
        append_manifest_entry(
            manifest_path,
            ReplayDatasetEntry("kaggle", "replays/episode_2.json", 1, "d2"),
        )


def test_append_manifest_entry_rejects_negative_new_size(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="negative"):
        append_manifest_entry(
            manifest_path,
            ReplayDatasetEntry("kaggle", "replays/episode_2.json", -1, "d2"),
        )


def test_append_manifest_entry_failed_replace_leaves_original_unchanged(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    original = [
        {
            "source": "kaggle",
            "local_path": "replays/episode_1.json",
            "size_bytes": 10,
            "discussion_id": "d1",
        }
    ]
    manifest_path.write_text(json.dumps(original), encoding="utf-8")

    entry = ReplayDatasetEntry("kaggle", "replays/episode_2.json", 20, "d2")

    with patch("tools.download_replays.os.replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError, match="replace failed"):
            append_manifest_entry(manifest_path, entry)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original


def test_parse_kaggle_files_table_parses_json_rows():
    text = """name,size,creationDate\nepisode_1.json,1.5MB,2026-05-01\nREADME.md,1KB,2026-05-01\nepisode_2.json,2 KB,2026-05-02\n"""
    rows = parse_kaggle_files_table(text)
    assert [r["name"] for r in rows] == ["episode_1.json", "README.md", "episode_2.json"]
    assert rows[0]["size_bytes"] == int(1.5 * 1024 * 1024)
    assert rows[2]["size_bytes"] == 2 * 1024


def test_parse_kaggle_files_table_parses_raw_integer_sizes():
    text = "name,size,creationDate\nepisode.json,2048,2026-05-01\n"
    rows = parse_kaggle_files_table(text)
    assert rows == [{"name": "episode.json", "size_bytes": 2048}]


def test_download_replays_follows_next_page_token_and_sets_page_size(tmp_path):
    output_dir = tmp_path / "replays"
    manifest_path = output_dir / "manifest.json"
    calls = []

    def fake_run(args, capture_output, text, check):
        calls.append(args)
        if args[:3] == ["kaggle", "datasets", "files"]:
            if "--page-token" not in args:
                class FirstPage:
                    stdout = "name,size\na.json,1KB\n\nNext Page Token = tok_2\n"
                    stderr = ""
                    returncode = 0
                return FirstPage()
            class SecondPage:
                stdout = "name,size\nb.json,1KB\n"
                stderr = ""
                returncode = 0
            return SecondPage()

        file_name = args[args.index("-f") + 1]
        (output_dir / file_name).parent.mkdir(parents=True, exist_ok=True)
        (output_dir / file_name).write_text("ok", encoding="utf-8")

        class DownloadResult:
            stdout = ""
            stderr = ""
            returncode = 0
        return DownloadResult()

    download_replays(100.0, 1.0, output_dir, manifest_path, run_command=fake_run, sources=[("dataset/name", "d1")])

    files_calls = [c for c in calls if c[:3] == ["kaggle", "datasets", "files"]]
    assert len(files_calls) == 2
    assert "--page-size" in files_calls[0]
    assert files_calls[0][files_calls[0].index("--page-size") + 1] == "100"
    assert "--page-token" in files_calls[1]
    assert files_calls[1][files_calls[1].index("--page-token") + 1] == "tok_2"


@pytest.mark.parametrize("hostile_name", ["../x.json", "/tmp/x.json", "episodes/../../x.json"])
def test_download_replays_rejects_hostile_file_names(tmp_path, hostile_name):
    output_dir = tmp_path / "replays"
    manifest_path = output_dir / "manifest.json"

    def fake_run(args, capture_output, text, check):
        if args[:3] == ["kaggle", "datasets", "files"]:
            class FilesResult:
                stdout = f"name,size\n{hostile_name},1KB\n"
                stderr = ""
                returncode = 0
            return FilesResult()
        raise AssertionError("download should not be called for hostile names")

    download_replays(100.0, 1.0, output_dir, manifest_path, run_command=fake_run, sources=[("dataset/name", "d1")])
    assert not manifest_path.exists()


def test_download_replays_raises_when_downloaded_artifact_missing(tmp_path):
    output_dir = tmp_path / "replays"
    manifest_path = output_dir / "manifest.json"

    def fake_run(args, capture_output, text, check):
        if args[:3] == ["kaggle", "datasets", "files"]:
            class FilesResult:
                stdout = "name,size\na.json,1KB\n"
                stderr = ""
                returncode = 0
            return FilesResult()
        class DownloadResult:
            stdout = ""
            stderr = ""
            returncode = 0
        return DownloadResult()

    with pytest.raises(FileNotFoundError, match="a.json"):
        download_replays(100.0, 1.0, output_dir, manifest_path, run_command=fake_run, sources=[("dataset/name", "d1")])
    assert not manifest_path.exists()


def test_download_replays_accepts_kaggle_flattened_download_path(tmp_path):
    output_dir = tmp_path / "replays"
    manifest_path = output_dir / "manifest.json"

    def fake_run(args, capture_output, text, check):
        if args[:3] == ["kaggle", "datasets", "files"]:
            class FilesResult:
                stdout = "name,size\nepisodes/episodes/flat.json,3\n"
                stderr = ""
                returncode = 0
            return FilesResult()
        (output_dir / "flat.json").write_text("{}", encoding="utf-8")
        class DownloadResult:
            stdout = ""
            stderr = ""
            returncode = 0
        return DownloadResult()

    download_replays(100.0, 1.0, output_dir, manifest_path, run_command=fake_run, sources=[("dataset/name", "d1")])

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data[0]["local_path"].endswith("flat.json")


def test_download_replays_respects_max_new_and_manifest_cap(tmp_path):
    output_dir = tmp_path / "replays"
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps([
        {"source": "kaggle", "local_path": "old.json", "size_bytes": 900, "discussion_id": "x"}
    ]), encoding="utf-8")

    calls = []

    def fake_run(args, capture_output, text, check):
        calls.append(args)
        if args[:3] == ["kaggle", "datasets", "files"]:
            class FilesResult:
                stdout = "name,size\na.json,800B\nb.json,500B\n"
                stderr = ""
                returncode = 0
            return FilesResult()

        file_name = args[args.index("-f") + 1]
        (output_dir / file_name).parent.mkdir(parents=True, exist_ok=True)
        (output_dir / file_name).write_text("ok", encoding="utf-8")
        class DownloadResult:
            stdout = ""
            stderr = ""
            returncode = 0
        return DownloadResult()

    download_replays(
        cap_gb=100.0,
        max_new_gb=1000 / (1024 ** 3),
        output_dir=output_dir,
        manifest_path=manifest_path,
        run_command=fake_run,
        sources=[("bovard/orbit-wars-top10-episodes-2026-05-04", "697413")],
    )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[-1]["local_path"].endswith("a.json")
    assert any(c[:3] == ["kaggle", "datasets", "download"] for c in calls)


def test_download_replays_missing_kaggle_executable_raises(tmp_path):
    import subprocess
    import pytest

    def fake_run(args, capture_output, text, check):
        raise FileNotFoundError("kaggle")

    with pytest.raises(SystemExit, match="kaggle"):
        download_replays(
            cap_gb=100.0,
            max_new_gb=0.1,
            output_dir=tmp_path / "replays",
            manifest_path=tmp_path / "replays" / "manifest.json",
            run_command=fake_run,
            sources=[("bovard/orbit-wars-top10-episodes-2026-05-04", "697413")],
        )


def test_download_replays_uses_csv_listing_flag_and_accepts_tar_gz(tmp_path):
    output_dir = tmp_path / "replays"
    manifest_path = output_dir / "manifest.json"
    calls = []

    def fake_run(args, capture_output, text, check):
        calls.append(args)
        if args[:3] == ["kaggle", "datasets", "files"]:
            class FilesResult:
                stdout = "name,size\nepisodes.tar.gz,2KB\n"
                stderr = ""
                returncode = 0

            return FilesResult()

        file_name = args[args.index("-f") + 1]
        (output_dir / file_name).parent.mkdir(parents=True, exist_ok=True)
        (output_dir / file_name).write_text("ok", encoding="utf-8")

        class DownloadResult:
            stdout = ""
            stderr = ""
            returncode = 0

        return DownloadResult()

    download_replays(
        cap_gb=100.0,
        max_new_gb=1.0,
        output_dir=output_dir,
        manifest_path=manifest_path,
        run_command=fake_run,
        sources=[("bovard/orbit-wars-top10-episodes-2026-05-04", "697413")],
    )

    files_calls = [c for c in calls if c[:3] == ["kaggle", "datasets", "files"]]
    assert files_calls
    assert "--csv" in files_calls[0]
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["local_path"].endswith("episodes.tar.gz")


@pytest.mark.parametrize("artifact_name", ["episode.json", "episodes.tar", "episodes.tar.gz", "episodes.tgz", "episodes.zip"])
def test_download_replays_accepts_supported_replay_artifact_extensions(tmp_path, artifact_name):
    output_dir = tmp_path / "replays"
    manifest_path = output_dir / "manifest.json"

    def fake_run(args, capture_output, text, check):
        if args[:3] == ["kaggle", "datasets", "files"]:
            class FilesResult:
                stdout = f"name,size\n{artifact_name},2KB\n"
                stderr = ""
                returncode = 0

            return FilesResult()

        file_name = args[args.index("-f") + 1]
        (output_dir / file_name).parent.mkdir(parents=True, exist_ok=True)
        (output_dir / file_name).write_text("ok", encoding="utf-8")

        class DownloadResult:
            stdout = ""
            stderr = ""
            returncode = 0

        return DownloadResult()

    download_replays(
        cap_gb=100.0,
        max_new_gb=1.0,
        output_dir=output_dir,
        manifest_path=manifest_path,
        run_command=fake_run,
        sources=[("bovard/orbit-wars-top10-episodes-2026-05-04", "697413")],
    )

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["local_path"].endswith(artifact_name)


def test_download_replays_script_entrypoint_does_not_raise_ordering_nameerror(tmp_path, monkeypatch):
    output_dir = tmp_path / "replays"
    manifest_path = output_dir / "manifest.json"

    def fake_run(args, capture_output, text, check):
        if args[:3] == ["kaggle", "datasets", "files"]:
            class FilesResult:
                stdout = "name,size\na.json,1KB\n"
                stderr = ""
                returncode = 0

            return FilesResult()

        file_name = args[args.index("-f") + 1]
        (output_dir / file_name).parent.mkdir(parents=True, exist_ok=True)
        (output_dir / file_name).write_text("ok", encoding="utf-8")

        class DownloadResult:
            stdout = ""
            stderr = ""
            returncode = 0

        return DownloadResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "download_replays.py",
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest_path),
            "--max-new-gb",
            "1.0",
            "--cap-gb",
            "100.0",
        ],
    )
    from tools import download_replays as module

    module.main()
    assert manifest_path.exists()


def test_download_replays_direct_main_entrypoint_does_not_raise_ordering_nameerror(tmp_path, monkeypatch):
    output_dir = tmp_path / "replays"
    manifest_path = output_dir / "manifest.json"

    def fake_run(args, capture_output, text, check):
        if args[:3] == ["kaggle", "datasets", "files"]:
            class FilesResult:
                stdout = "name,size\na.json,1KB\n"
                stderr = ""
                returncode = 0

            return FilesResult()

        file_name = args[args.index("-f") + 1]
        (output_dir / file_name).parent.mkdir(parents=True, exist_ok=True)
        (output_dir / file_name).write_text("ok", encoding="utf-8")

        class DownloadResult:
            stdout = ""
            stderr = ""
            returncode = 0

        return DownloadResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "download_replays.py",
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest_path),
            "--max-new-gb",
            "1.0",
            "--cap-gb",
            "100.0",
        ],
    )

    runpy.run_path("/home/chris/orbitwars/tools/download_replays.py", run_name="__main__")
    assert manifest_path.exists()
