#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from generate_release_manifest import ARTIFACTS, generate
from release_contract import ContractError
from release_publication_state import (
    APPROVED_ASSETS_REPOSITORY,
    CandidateIdentity,
    PUBLICATION_FILES,
    classify,
    publication_state_changed,
    require_approved_assets_repo,
    validate_candidate,
)


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


class ReleasePublicationStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.repository = root / "assets"
        self.candidate = root / "candidate"
        self.repository.mkdir()
        self.candidate.mkdir()
        git(self.repository, "init", "-q")
        git(self.repository, "config", "user.name", "test")
        git(self.repository, "config", "user.email", "test@example.com")

        for index, name in enumerate(ARTIFACTS):
            (self.candidate / name).write_bytes(f"candidate-{index}".encode())
        self.identity = CandidateIdentity(
            release_tag="v0.2.0",
            release_rebuild=0,
            assets_repo=APPROVED_ASSETS_REPOSITORY,
            bridge_commit="a" * 40,
            upstream_tag="v0.2.0",
            upstream_commit="b" * 40,
            native_release_tag="v0.2.0",
            native_manifest_sha256="c" * 64,
            native_commit="d" * 40,
            emscripten_version="6.0.8",
        )
        generate(argparse.Namespace(
            out_dir=self.candidate,
            release_tag="v0.2.0",
            release_rebuild=0,
            assets_repo=APPROVED_ASSETS_REPOSITORY,
            bridge_repo="leehack/llama-web-bridge",
            bridge_commit="a" * 40,
            upstream_repo="ggml-org/llama.cpp",
            upstream_tag="v0.2.0",
            upstream_commit="b" * 40,
            native_repo="leehack/llamadart-native",
            native_release_tag="v0.2.0",
            native_manifest_sha256="c" * 64,
            native_commit="d" * 40,
            emscripten_version="6.0.8",
        ))
        previous = {
            "bridge_assets_tag": "v0.1.23",
            "llama_cpp_tag": "b10514",
            "source_commit": "e" * 40,
        }
        (self.repository / "manifest.json").write_text(json.dumps(previous) + "\n")
        (self.repository / "README.md").write_text("assets\n")
        git(self.repository, "add", ".")
        git(self.repository, "commit", "-q", "-m", "previous")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @property
    def head(self) -> str:
        return git(self.repository, "rev-parse", "HEAD")

    def release_data(self, commit: str) -> dict[str, object]:
        fingerprint = validate_candidate(self.candidate, self.identity)
        assets = []
        for name in PUBLICATION_FILES:
            data = (self.candidate / name).read_bytes()
            assets.append({
                "name": name,
                "state": "uploaded",
                "size": len(data),
                "digest": f"sha256:{__import__('hashlib').sha256(data).hexdigest()}",
            })
        return {
            "id": 42,
            "tag_name": "v0.2.0",
            "name": "v0.2.0",
            "draft": False,
            "prerelease": False,
            "target_commitish": commit,
            "body": f"Candidate fingerprint: `{fingerprint}`",
            "assets": assets,
        }

    def inspect(
        self,
        *,
        tag: str | None = None,
        release: bool | dict[str, object] = False,
    ) -> dict[str, object]:
        release_data = self.release_data(tag or self.head) if release is True else release or None
        return classify(
            repository=self.repository,
            candidate=self.candidate,
            identity=self.identity,
            branch_commit=self.head,
            tag_commit=tag,
            release=release_data,
        )

    def publish_branch_candidate(self) -> str:
        for path in self.candidate.iterdir():
            shutil.copy2(path, self.repository / path.name)
        git(self.repository, "add", ".")
        git(self.repository, "commit", "-q", "-m", "candidate")
        return self.head

    def test_target_is_locked_before_credentials(self) -> None:
        self.assertEqual(
            require_approved_assets_repo(APPROVED_ASSETS_REPOSITORY),
            APPROVED_ASSETS_REPOSITORY,
        )
        with self.assertRaises(ContractError):
            require_approved_assets_repo("attacker/unrelated-assets")

    def test_no_refs_is_new_publication(self) -> None:
        state = self.inspect()
        self.assertEqual(state["state"], "absent")
        self.assertEqual(state["action"], "publish-refs-and-release")
        self.assertEqual(state["outcome"], "newly-published")

    def test_branch_only_is_safely_resumable(self) -> None:
        self.publish_branch_candidate()
        state = self.inspect()
        self.assertEqual(state["state"], "branch-only")
        self.assertEqual(state["action"], "publish-tag-and-release")
        self.assertEqual(state["outcome"], "safely-resumed")

    def test_same_source_rebuild_may_change_only_manifest(self) -> None:
        self.publish_branch_candidate()
        rebuild = CandidateIdentity(
            release_tag="v0.2.0-1",
            release_rebuild=1,
            assets_repo=APPROVED_ASSETS_REPOSITORY,
            bridge_commit=self.identity.bridge_commit,
            upstream_tag=self.identity.upstream_tag,
            upstream_commit=self.identity.upstream_commit,
            native_release_tag=self.identity.native_release_tag,
            native_manifest_sha256=self.identity.native_manifest_sha256,
            native_commit=self.identity.native_commit,
            emscripten_version=self.identity.emscripten_version,
        )
        generate(argparse.Namespace(
            out_dir=self.candidate,
            release_tag=rebuild.release_tag,
            release_rebuild=rebuild.release_rebuild,
            assets_repo=APPROVED_ASSETS_REPOSITORY,
            bridge_repo="leehack/llama-web-bridge",
            bridge_commit=rebuild.bridge_commit,
            upstream_repo="ggml-org/llama.cpp",
            upstream_tag=rebuild.upstream_tag,
            upstream_commit=rebuild.upstream_commit,
            native_repo="leehack/llamadart-native",
            native_release_tag=rebuild.native_release_tag,
            native_manifest_sha256=rebuild.native_manifest_sha256,
            native_commit=rebuild.native_commit,
            emscripten_version=rebuild.emscripten_version,
        ))
        shutil.copy2(self.candidate / "manifest.json", self.repository / "manifest.json")
        git(self.repository, "add", "manifest.json")
        git(self.repository, "commit", "-q", "-m", "rebuild")
        state = classify(
            repository=self.repository,
            candidate=self.candidate,
            identity=rebuild,
            branch_commit=self.head,
            tag_commit=None,
            release=None,
        )
        self.assertTrue(state["allowed"])
        self.assertEqual(state["state"], "branch-only")

    def test_post_push_release_failure_retries_then_completes(self) -> None:
        candidate_commit = self.publish_branch_candidate()
        git(self.repository, "tag", "v0.2.0", candidate_commit)

        retry = self.inspect(tag=candidate_commit, release=False)
        self.assertEqual(retry["state"], "tag-without-release")
        self.assertEqual(retry["action"], "create-release")
        self.assertEqual(retry["outcome"], "safely-resumed")

        complete = self.inspect(tag=candidate_commit, release=True)
        self.assertEqual(complete["state"], "complete")
        self.assertEqual(complete["action"], "none")
        self.assertEqual(complete["outcome"], "already-complete")
        self.assertEqual(complete["release_id"], 42)

    def test_rejected_and_ambiguous_push_mutation_outcomes_are_exact(self) -> None:
        before = self.inspect()
        unchanged = self.inspect()
        self.assertFalse(publication_state_changed(before, unchanged))

        candidate_commit = self.publish_branch_candidate()
        git(self.repository, "tag", "v0.2.0", candidate_commit)
        accepted_but_client_failed = self.inspect(tag=candidate_commit, release=False)
        self.assertTrue(publication_state_changed(before, accepted_but_client_failed))

    def test_ambiguous_release_create_failure_is_recovered_after_exact_requery(self) -> None:
        candidate_commit = self.publish_branch_candidate()
        git(self.repository, "tag", "v0.2.0", candidate_commit)
        remote_release: dict[str, object] | None = None
        mutation_calls = 0

        before = self.inspect(tag=candidate_commit, release=remote_release or False)
        self.assertEqual(before["action"], "create-release")
        mutation_calls += 1
        remote_release = self.release_data(candidate_commit)
        # Simulate the server committing the release and the client then failing.
        after = self.inspect(tag=candidate_commit, release=remote_release)
        self.assertEqual(after["outcome"], "already-complete")
        self.assertEqual(mutation_calls, 1)

    def test_release_create_failure_without_server_commit_remains_retryable(self) -> None:
        candidate_commit = self.publish_branch_candidate()
        git(self.repository, "tag", "v0.2.0", candidate_commit)
        after = self.inspect(tag=candidate_commit, release=False)
        self.assertEqual(after["reason_code"], "exact-tag-release-missing")
        self.assertTrue(after["retryable"])

    def test_partial_release_asset_upload_is_safely_resumable(self) -> None:
        candidate_commit = self.publish_branch_candidate()
        git(self.repository, "tag", "v0.2.0", candidate_commit)
        partial_release = self.release_data(candidate_commit)
        missing = partial_release["assets"].pop()
        partial = self.inspect(tag=candidate_commit, release=partial_release)
        self.assertTrue(partial["allowed"])
        self.assertEqual(partial["state"], "release-assets-partial")
        self.assertEqual(partial["action"], "upload-release-assets")
        self.assertEqual(partial["missing_release_assets"], [missing["name"]])
        complete = self.inspect(tag=candidate_commit, release=True)
        self.assertEqual(complete["outcome"], "already-complete")

    def test_exact_complete_release_remains_idempotent_after_branch_advances(self) -> None:
        candidate_commit = self.publish_branch_candidate()
        git(self.repository, "tag", "v0.2.0", candidate_commit)
        (self.repository / "README.md").write_text("later release\n")
        git(self.repository, "add", "README.md")
        git(self.repository, "commit", "-q", "-m", "later")
        state = self.inspect(tag=candidate_commit, release=True)
        self.assertEqual(state["outcome"], "already-complete")

    def test_existing_tag_mismatch_fails_closed_as_collision(self) -> None:
        wrong_commit = self.head
        state = self.inspect(tag=wrong_commit)
        self.assertFalse(state["allowed"])
        self.assertEqual(state["outcome"], "collision")

    def test_unreachable_exact_tag_fails_closed(self) -> None:
        candidate_commit = self.publish_branch_candidate()
        previous_branch = git(self.repository, "rev-parse", f"{candidate_commit}^")
        git(self.repository, "checkout", "-q", "--detach", previous_branch)
        for path in self.candidate.iterdir():
            shutil.copy2(path, self.repository / path.name)
        git(self.repository, "add", ".")
        git(self.repository, "commit", "-q", "-m", "orphan-candidate")
        orphan = self.head
        git(self.repository, "tag", "v0.2.0", orphan)
        state = classify(
            repository=self.repository,
            candidate=self.candidate,
            identity=self.identity,
            branch_commit=candidate_commit,
            tag_commit=orphan,
            release=None,
        )
        self.assertFalse(state["allowed"])
        self.assertIn("not reachable", state["reason"])

    def test_branch_only_with_unrelated_diff_fails_closed(self) -> None:
        for path in self.candidate.iterdir():
            shutil.copy2(path, self.repository / path.name)
        (self.repository / "UNEXPECTED").write_text("injected\n")
        git(self.repository, "add", ".")
        git(self.repository, "commit", "-q", "-m", "candidate-plus-extra")
        state = self.inspect()
        self.assertFalse(state["allowed"])
        self.assertIn("governed release files", state["reason"])

    def test_release_without_tag_fails_closed(self) -> None:
        state = self.inspect(release=True)
        self.assertFalse(state["allowed"])
        self.assertEqual(state["outcome"], "collision")

    def test_release_metadata_and_asset_digest_mismatch_fail_closed(self) -> None:
        candidate_commit = self.publish_branch_candidate()
        git(self.repository, "tag", "v0.2.0", candidate_commit)
        for field, value in (("prerelease", True), ("target_commitish", "f" * 40)):
            release = self.release_data(candidate_commit)
            release[field] = value
            with self.subTest(field=field):
                self.assertFalse(self.inspect(tag=candidate_commit, release=release)["allowed"])
        release = self.release_data(candidate_commit)
        release["assets"][0]["digest"] = "sha256:" + "0" * 64
        self.assertFalse(self.inspect(tag=candidate_commit, release=release)["allowed"])
        release = self.release_data(candidate_commit)
        release["assets"].append({
            "name": "unexpected.bin",
            "state": "uploaded",
            "size": 0,
            "digest": "sha256:" + hashlib.sha256(b"").hexdigest(),
        })
        self.assertFalse(self.inspect(tag=candidate_commit, release=release)["allowed"])

    def test_candidate_checksum_mismatch_fails_closed(self) -> None:
        (self.candidate / ARTIFACTS[0]).write_bytes(b"tampered")
        with self.assertRaises(ContractError):
            classify(
                repository=self.repository,
                candidate=self.candidate,
                identity=self.identity,
                branch_commit=self.head,
                tag_commit=None,
                release=None,
            )

    def test_rollback_has_machine_readable_outcome(self) -> None:
        newer = json.loads((self.repository / "manifest.json").read_text())
        newer["bridge_assets_tag"] = "v0.3.0"
        newer["llama_cpp_tag"] = "v0.3.0"
        (self.repository / "manifest.json").write_text(json.dumps(newer) + "\n")
        git(self.repository, "add", "manifest.json")
        git(self.repository, "commit", "-q", "-m", "newer")
        state = self.inspect()
        self.assertFalse(state["allowed"])
        self.assertEqual(state["state"], "rollback")
        self.assertEqual(state["outcome"], "rollback")

    def test_new_upstream_rebuild_is_machine_readable_rollback(self) -> None:
        identity = CandidateIdentity(
            release_tag="v0.2.1-1",
            release_rebuild=1,
            assets_repo=APPROVED_ASSETS_REPOSITORY,
            bridge_commit=self.identity.bridge_commit,
            upstream_tag="v0.2.1",
            upstream_commit=self.identity.upstream_commit,
            native_release_tag="v0.2.1-1",
            native_manifest_sha256=self.identity.native_manifest_sha256,
            native_commit=self.identity.native_commit,
            emscripten_version=self.identity.emscripten_version,
        )
        generate(argparse.Namespace(
            out_dir=self.candidate,
            release_tag=identity.release_tag,
            release_rebuild=identity.release_rebuild,
            assets_repo=APPROVED_ASSETS_REPOSITORY,
            bridge_repo="leehack/llama-web-bridge",
            bridge_commit=identity.bridge_commit,
            upstream_repo="ggml-org/llama.cpp",
            upstream_tag=identity.upstream_tag,
            upstream_commit=identity.upstream_commit,
            native_repo="leehack/llamadart-native",
            native_release_tag=identity.native_release_tag,
            native_manifest_sha256=identity.native_manifest_sha256,
            native_commit=identity.native_commit,
            emscripten_version=identity.emscripten_version,
        ))
        state = classify(
            repository=self.repository,
            candidate=self.candidate,
            identity=identity,
            branch_commit=self.head,
            tag_commit=None,
            release=None,
        )
        self.assertEqual(state["outcome"], "rollback")
        self.assertEqual(state["reason_code"], "ordering-rollback")


if __name__ == "__main__":
    unittest.main()
