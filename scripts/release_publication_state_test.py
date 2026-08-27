#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from generate_release_manifest import (
    ARTIFACTS,
    LOCAL_ATTESTATION_REQUIRED,
    generate,
)
from release_contract import ContractError
from release_contract_test import release_attestation
from release_publication_state import (
    APPROVED_ASSETS_REPOSITORY,
    CandidateIdentity,
    PUBLICATION_FILES,
    _fatal,
    candidate_publication_digests,
    classify,
    mutation_unknown_from_requery,
    mutation_unknown_outcome,
    publication_state_changed,
    require_approved_assets_repo,
    validate_candidate,
    verify_immutable_publication,
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
            upstream_commit="bb4caa7540188872173c44d161602d9271386413",
            native_release_tag="v0.2.0",
            native_manifest_sha256="2e5d29d7f98f0d71e75d3fa63b7c55f3b2a7933247cc34ea2b1c5e053d142452",
            native_commit="e5c240e34b525da953ed98dc743516eef78cb738",
            emscripten_version="6.0.8",
            orchestrator_correlation_id="kanban:t_fcc0b814:web-v0.1.38",
            github_run_id="123456789",
            github_run_url="https://github.com/leehack/llama-web-bridge/actions/runs/123456789",
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
            upstream_commit=self.identity.upstream_commit,
            native_repo="leehack/llamadart-native",
            native_release_tag="v0.2.0",
            native_manifest_sha256=self.identity.native_manifest_sha256,
            native_commit=self.identity.native_commit,
            emscripten_version="6.0.8",
            orchestrator_correlation_id=self.identity.orchestrator_correlation_id,
            github_run_id=self.identity.github_run_id,
            github_run_url=self.identity.github_run_url,
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

    def published_release(self, **overrides: object) -> dict[str, object]:
        release = {
            **self.release_data(self.head),
            "immutable": True,
        }
        release.update(overrides)
        return release

    def published_attestation(self, **overrides: object) -> dict[str, object]:
        return release_attestation(
            release_tag="v0.2.0",
            assets_repo=APPROVED_ASSETS_REPOSITORY,
            tag_commit=self.head,
            release_id=42,
            assets=candidate_publication_digests(self.candidate),
            **overrides,
        )

    def verify_published(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "candidate": self.candidate,
            "assets_repo": APPROVED_ASSETS_REPOSITORY,
            "release_tag": "v0.2.0",
            "tag_commit": self.head,
            "release_id": 42,
            "release_by_tag": self.published_release(),
            "release_by_id": self.published_release(),
            "attestation": self.published_attestation(),
        }
        arguments.update(overrides)
        return verify_immutable_publication(**arguments)

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
            "published_at": "2026-08-20T22:15:59Z",
            "body": (
                f"Candidate fingerprint: `{fingerprint}`\n"
                f"Orchestrator correlation: `{self.identity.orchestrator_correlation_id}`\n"
                f"Bridge workflow run: {self.identity.github_run_url}"
            ),
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

    def commit_history_manifest(self, release_tag: str, upstream_tag: str) -> None:
        manifest = {
            "bridge_assets_tag": release_tag,
            "llama_cpp_tag": upstream_tag,
            "source_commit": "e" * 40,
        }
        (self.repository / "manifest.json").write_text(json.dumps(manifest) + "\n")
        git(self.repository, "add", "manifest.json")
        git(self.repository, "commit", "-q", "-m", f"history {release_tag}")

    def regenerate_candidate(
        self,
        release_tag: str,
        release_rebuild: int,
        upstream_tag: str,
        native_release_tag: str | None = None,
    ) -> CandidateIdentity:
        identity = CandidateIdentity(
            release_tag=release_tag,
            release_rebuild=release_rebuild,
            assets_repo=APPROVED_ASSETS_REPOSITORY,
            bridge_commit=self.identity.bridge_commit,
            upstream_tag=upstream_tag,
            upstream_commit=self.identity.upstream_commit,
            native_release_tag=native_release_tag or release_tag,
            native_manifest_sha256=self.identity.native_manifest_sha256,
            native_commit=self.identity.native_commit,
            emscripten_version=self.identity.emscripten_version,
            orchestrator_correlation_id=self.identity.orchestrator_correlation_id,
            github_run_id=self.identity.github_run_id,
            github_run_url=self.identity.github_run_url,
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
            orchestrator_correlation_id=identity.orchestrator_correlation_id,
            github_run_id=identity.github_run_id,
            github_run_url=identity.github_run_url,
        ))
        return identity

    def classify_identity(self, identity: CandidateIdentity) -> dict[str, object]:
        return classify(
            repository=self.repository,
            candidate=self.candidate,
            identity=identity,
            branch_commit=self.head,
            tag_commit=None,
            release=None,
        )

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

    def test_development_advances_from_current_legacy_stable_tag(self) -> None:
        self.commit_history_manifest("v0.1.37", "b10514")
        identity = self.regenerate_candidate("b10515", 0, "b10515")
        state = self.classify_identity(identity)
        self.assertTrue(state["allowed"])
        self.assertEqual(state["action"], "publish-refs-and-release")

    def test_development_history_rejects_rollback_and_collision(self) -> None:
        self.commit_history_manifest("b10514", "b10514")
        rollback = self.regenerate_candidate("b10513", 0, "b10513")
        rollback_state = self.classify_identity(rollback)
        self.assertEqual(rollback_state["outcome"], "rollback")
        collision = self.regenerate_candidate("b10514", 0, "b10514")
        collision_state = self.classify_identity(collision)
        self.assertEqual(collision_state["outcome"], "collision")

    def test_development_rebuild_advances_on_its_own_release_line(self) -> None:
        self.commit_history_manifest("b10514", "b10514")
        identity = self.regenerate_candidate("b10514-1", 1, "b10514")
        state = self.classify_identity(identity)
        self.assertTrue(state["allowed"])

    def test_first_release_in_a_channel_must_use_rebuild_zero(self) -> None:
        """A rebuild of an asset tag never published in this channel is fail-closed."""
        self.commit_history_manifest("v0.1.37", "b10514")
        identity = self.regenerate_candidate("b10514-1", 1, "b10514")
        state = self.classify_identity(identity)
        self.assertFalse(state["allowed"])
        self.assertEqual(state["outcome"], "rollback")

    def test_development_rebuild_history_rejects_rollback_and_collision(self) -> None:
        self.commit_history_manifest("b10514-2", "b10514")
        rollback = self.regenerate_candidate("b10514-1", 1, "b10514")
        rollback_state = self.classify_identity(rollback)
        self.assertEqual(rollback_state["outcome"], "rollback")
        collision = self.regenerate_candidate("b10514-2", 2, "b10514")
        collision_state = self.classify_identity(collision)
        self.assertEqual(collision_state["outcome"], "collision")

    def test_chief_candidate_advances_independent_release_and_upstream_lines(self) -> None:
        """v0.1.38/v0.2.0 after v0.1.37/b10514: assets advance, upstream migrates."""
        self.commit_history_manifest("v0.1.37", "b10514")
        identity = self.regenerate_candidate("v0.1.38", 0, "v0.2.0", "v0.2.0-1")
        state = self.classify_identity(identity)
        self.assertTrue(state["allowed"], state["reason"])
        self.assertEqual(state["state"], "absent")
        self.assertEqual(state["action"], "publish-refs-and-release")
        self.assertEqual(state["outcome"], "newly-published")
        self.assertEqual(state["release_tag"], "v0.1.38")

    def test_schema_v2_stores_independent_release_and_upstream_identities(self) -> None:
        identity = self.regenerate_candidate("v0.1.38", 0, "v0.2.0", "v0.2.0-1")
        self.assertIsInstance(validate_candidate(self.candidate, identity), str)
        manifest = json.loads((self.candidate / "manifest.json").read_text())
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["release_tag"], "v0.1.38")
        self.assertEqual(manifest["upstream_tag"], "v0.2.0")
        self.assertEqual(manifest["release_channel"], "stable")
        self.assertEqual(manifest["release_rebuild"], 0)
        self.assertEqual(manifest["native_release_tag"], "v0.2.0-1")
        # Legacy aliases must mirror their own field, never the other identity.
        self.assertEqual(manifest["bridge_assets_tag"], "v0.1.38")
        self.assertEqual(manifest["llama_cpp_tag"], "v0.2.0")

    def test_schema_v2_history_is_read_without_fabricating_identities(self) -> None:
        self.regenerate_candidate("v0.1.38", 0, "v0.2.0", "v0.2.0-1")
        shutil.copy2(self.candidate / "manifest.json", self.repository / "manifest.json")
        git(self.repository, "add", "manifest.json")
        git(self.repository, "commit", "-q", "-m", "schema-v2 history")

        advance = self.regenerate_candidate("v0.1.39", 0, "v0.2.1", "v0.2.1")
        self.assertTrue(self.classify_identity(advance)["allowed"])
        # Ordering uses the recorded v0.1.38, not the recorded upstream v0.2.0.
        release_rollback = self.regenerate_candidate("v0.1.37", 0, "v0.2.0", "v0.2.0-1")
        self.assertEqual(self.classify_identity(release_rollback)["outcome"], "rollback")
        # ...and the upstream dimension uses the recorded v0.2.0, not v0.1.38.
        upstream_rollback = self.regenerate_candidate(
            "v0.1.39", 0, "v0.1.9", "v0.1.9"
        )
        upstream_state = self.classify_identity(upstream_rollback)
        self.assertEqual(upstream_state["outcome"], "rollback")
        self.assertIn("upstream transition is backward", upstream_state["reason"])

    def test_conflicting_history_aliases_fail_closed(self) -> None:
        for field, value in (
            ("bridge_assets_tag", "v0.1.37"),
            ("llama_cpp_tag", "b10514"),
        ):
            manifest = {
                "release_tag": "v0.1.38",
                "bridge_assets_tag": "v0.1.38",
                "upstream_tag": "v0.2.0",
                "llama_cpp_tag": "v0.2.0",
            }
            manifest[field] = value
            (self.repository / "manifest.json").write_text(json.dumps(manifest) + "\n")
            git(self.repository, "add", "manifest.json")
            git(self.repository, "commit", "-q", "-m", f"conflicting {field}")
            identity = self.regenerate_candidate(
                "v0.1.39", 0, "v0.2.1", "v0.2.1"
            )
            with self.subTest(field=field):
                state = self.classify_identity(identity)
                self.assertFalse(state["allowed"])
                self.assertEqual(state["outcome"], "collision")
                self.assertIn("aliases conflict", state["reason"])

    def test_independent_release_rollback_and_collision_are_rejected(self) -> None:
        self.commit_history_manifest("v0.1.38", "v0.2.0")
        rollback = self.regenerate_candidate("v0.1.37", 0, "v0.2.0", "v0.2.0-1")
        rollback_state = self.classify_identity(rollback)
        self.assertEqual(rollback_state["outcome"], "rollback")
        self.assertFalse(rollback_state["allowed"])
        collision = self.regenerate_candidate("v0.1.38", 0, "v0.2.0", "v0.2.0-1")
        collision_state = self.classify_identity(collision)
        self.assertEqual(collision_state["outcome"], "collision")
        self.assertFalse(collision_state["allowed"])

    def test_independent_upstream_rollback_is_rejected(self) -> None:
        """The asset tag advances legally, but the upstream line must not regress."""
        self.commit_history_manifest("v0.1.38", "v0.2.1")
        identity = self.regenerate_candidate("v0.1.39", 0, "v0.2.0", "v0.2.0-1")
        state = self.classify_identity(identity)
        self.assertFalse(state["allowed"])
        self.assertEqual(state["outcome"], "rollback")
        self.assertIn("upstream transition is backward", state["reason"])

    def test_stable_to_development_upstream_is_forbidden(self) -> None:
        self.commit_history_manifest("v0.1.38", "v0.2.0")
        identity = self.regenerate_candidate("v0.1.39", 0, "b10600", "b10600")
        state = self.classify_identity(identity)
        self.assertFalse(state["allowed"])
        self.assertEqual(state["outcome"], "rollback")
        self.assertIn(
            "upstream transition is forbidden-stable-to-development", state["reason"]
        )

    def test_new_release_version_must_restart_rebuild_numbering(self) -> None:
        self.commit_history_manifest("v0.1.37", "b10514")
        identity = self.regenerate_candidate("v0.1.38-1", 1, "v0.2.0", "v0.2.0-1")
        state = self.classify_identity(identity)
        self.assertFalse(state["allowed"])
        self.assertEqual(state["outcome"], "rollback")

    def test_release_rebuild_must_match_the_release_tag(self) -> None:
        identity = self.regenerate_candidate("v0.1.38", 0, "v0.2.0", "v0.2.0-1")
        mismatched = CandidateIdentity(
            **{**identity.__dict__, "release_rebuild": 1}
        )
        with self.assertRaises(ContractError):
            validate_candidate(self.candidate, mismatched)

    def test_candidate_upstream_tag_syntax_stays_exact(self) -> None:
        identity = self.regenerate_candidate("v0.1.38", 0, "v0.2.0", "v0.2.0-1")
        for invalid in ("main", "v0.2", "v0.2.0-1", "b10514-1", ""):
            with self.subTest(upstream_tag=invalid):
                broken = CandidateIdentity(
                    **{**identity.__dict__, "upstream_tag": invalid}
                )
                with self.assertRaises(ContractError):
                    validate_candidate(self.candidate, broken)

    def test_native_release_tag_must_still_encode_its_upstream(self) -> None:
        """Bridge assets decouple from upstream; native releases never do."""
        accepted = self.regenerate_candidate(
            "v0.1.38", 0, "v0.2.0", "v0.2.0-1"
        )
        self.assertIsInstance(validate_candidate(self.candidate, accepted), str)
        for native_tag, upstream_tag in (
            ("v0.1.38", "v0.2.0"),
            ("v0.2.1-1", "v0.2.0"),
            ("b10514", "v0.2.0"),
            ("b10515", "b10514"),
        ):
            with self.subTest(native_tag=native_tag, upstream_tag=upstream_tag):
                identity = self.regenerate_candidate(
                    "v0.1.38", 0, upstream_tag, native_tag
                )
                with self.assertRaises(ContractError):
                    validate_candidate(self.candidate, identity)

    def test_stable_history_ignores_later_development_publication(self) -> None:
        self.commit_history_manifest("v0.2.0", "v0.2.0")
        self.commit_history_manifest("b10515", "b10515")
        identity = self.regenerate_candidate("v0.2.1", 0, "v0.2.1")
        state = self.classify_identity(identity)
        self.assertTrue(state["allowed"])

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
            orchestrator_correlation_id=self.identity.orchestrator_correlation_id,
            github_run_id=self.identity.github_run_id,
            github_run_url=self.identity.github_run_url,
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
            orchestrator_correlation_id=rebuild.orchestrator_correlation_id,
            github_run_id=rebuild.github_run_id,
            github_run_url=rebuild.github_run_url,
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

    def test_empty_requery_after_ambiguous_push_is_retryable_mutation_unknown(self) -> None:
        empty_requery = self.candidate.parent / "empty-requery.json"
        empty_requery.write_bytes(b"")
        outcome = mutation_unknown_from_requery(
            self.candidate,
            self.identity,
            "ref-requery-failed",
            empty_requery,
        )
        self.assertEqual(outcome["state"], "mutation-unknown")
        self.assertEqual(outcome["outcome"], "mutation-unknown")
        self.assertEqual(outcome["reason_code"], "ref-requery-failed")
        self.assertTrue(outcome["retryable"])
        self.assertIsNone(outcome["mutated"])
        self.assertEqual(outcome["mutation_status"], "unknown")
        self.assertEqual(
            outcome["orchestrator_correlation_id"],
            "kanban:t_fcc0b814:web-v0.1.38",
        )
        self.assertEqual(outcome["github_run_id"], "123456789")
        self.assertEqual(
            outcome["qualification_gates"]["text_to_speech"],
            LOCAL_ATTESTATION_REQUIRED,
        )
        with self.assertRaises(ContractError):
            mutation_unknown_outcome(self.candidate, self.identity, "attacker-value")
        empty_requery.write_text(json.dumps({"schema_version": 1, "state": "exact"}))
        with self.assertRaises(ContractError):
            mutation_unknown_from_requery(
                self.candidate,
                self.identity,
                "ref-requery-failed",
                empty_requery,
            )

    def test_empty_release_requery_is_retryable_mutation_unknown(self) -> None:
        empty_requery = self.candidate.parent / "empty-release-requery.json"
        empty_requery.write_bytes(b"")
        outcome = mutation_unknown_from_requery(
            self.candidate,
            self.identity,
            "release-requery-failed",
            empty_requery,
        )
        self.assertEqual(outcome["state"], "mutation-unknown")
        self.assertEqual(outcome["reason_code"], "release-requery-failed")
        self.assertTrue(outcome["retryable"])
        self.assertIsNone(outcome["mutated"])

    def test_fatal_classifier_requery_is_not_treated_as_exact_remote_state(self) -> None:
        fatal_requery = self.candidate.parent / "fatal-requery.json"
        fatal_requery.write_text(
            json.dumps(_fatal("empty branch identity", self.identity))
        )
        outcome = mutation_unknown_from_requery(
            self.candidate,
            self.identity,
            "ref-requery-failed",
            fatal_requery,
        )
        self.assertEqual(outcome["state"], "mutation-unknown")
        self.assertTrue(outcome["retryable"])
        self.assertIsNone(outcome["mutated"])

    def test_fatal_outcome_preserves_run_and_qualification_provenance(self) -> None:
        outcome = _fatal("invalid candidate", self.identity)
        self.assertEqual(
            outcome["orchestrator_correlation_id"],
            self.identity.orchestrator_correlation_id,
        )
        self.assertEqual(outcome["github_run_id"], self.identity.github_run_id)
        self.assertEqual(outcome["github_run_url"], self.identity.github_run_url)
        self.assertEqual(outcome["qualification_gates"]["state_persistence"], "passed")
        self.assertEqual(outcome["qualification_gates"]["multimodal"], "passed")
        self.assertEqual(
            outcome["qualification_gates"]["speech_to_text"],
            LOCAL_ATTESTATION_REQUIRED,
        )
        self.assertEqual(
            outcome["qualification_gates"]["text_to_speech"],
            LOCAL_ATTESTATION_REQUIRED,
        )

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

    def test_partial_published_release_is_not_repaired(self) -> None:
        candidate_commit = self.publish_branch_candidate()
        git(self.repository, "tag", "v0.2.0", candidate_commit)
        partial_release = self.release_data(candidate_commit)
        missing = partial_release["assets"].pop()
        partial = self.inspect(tag=candidate_commit, release=partial_release)
        self.assertFalse(partial["allowed"])
        self.assertEqual(partial["state"], "release-assets-partial")
        self.assertEqual(partial["action"], "none")
        self.assertEqual(partial["outcome"], "immutable-publication-unverified")
        self.assertFalse(partial["retryable"])
        self.assertEqual(partial["missing_release_assets"], [missing["name"]])

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
        for field, value in (
            ("prerelease", True),
            ("target_commitish", "f" * 40),
            ("id", True),
            ("id", 0),
        ):
            release = self.release_data(candidate_commit)
            release[field] = value
            with self.subTest(field=field):
                self.assertFalse(self.inspect(tag=candidate_commit, release=release)["allowed"])
        release = self.release_data(candidate_commit)
        release["assets"][0]["digest"] = "sha256:" + "0" * 64
        self.assertFalse(self.inspect(tag=candidate_commit, release=release)["allowed"])
        release = self.release_data(candidate_commit)
        release["body"] = release["body"].replace(
            self.identity.orchestrator_correlation_id, "different-run"
        )
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

    def test_candidate_inventory_must_be_exact(self) -> None:
        (self.candidate / "unexpected.bin").write_bytes(b"not governed")
        with self.assertRaises(ContractError):
            validate_candidate(self.candidate, self.identity)

    def test_candidate_manifest_duplicate_keys_fail_closed(self) -> None:
        manifest_path = self.candidate / "manifest.json"
        original = manifest_path.read_text(encoding="utf-8")
        duplicate = original.replace(
            "{\n", f'{{\n  "release_tag": "{self.identity.release_tag}",\n', 1
        )
        manifest_path.write_text(duplicate, encoding="utf-8")
        with self.assertRaises(ContractError):
            validate_candidate(self.candidate, self.identity)

    def test_candidate_symlink_is_not_an_immutable_regular_file(self) -> None:
        artifact = self.candidate / ARTIFACTS[0]
        target = Path(self.temporary.name) / "outside-artifact"
        target.write_bytes(artifact.read_bytes())
        artifact.unlink()
        artifact.symlink_to(target)
        with self.assertRaises(ContractError):
            validate_candidate(self.candidate, self.identity)
        with self.assertRaises(ContractError):
            candidate_publication_digests(self.candidate)

    def test_candidate_run_and_gate_provenance_tampering_fails_closed(self) -> None:
        for field, value in (
            ("orchestrator_correlation_id", "different-run"),
            ("github_run_url", "https://github.com/attacker/repo/actions/runs/123"),
            ("qualification_gates", {"state_persistence": "passed"}),
            (
                "qualification_gates",
                {
                    "state_persistence": "passed",
                    "multimodal": "passed",
                    "speech_to_text": "passed",
                    "text_to_speech": "passed",
                },
            ),
            ("unproven_capabilities", {"real_device_playback": "proven"}),
        ):
            manifest_path = self.candidate / "manifest.json"
            original = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(original)
            manifest[field] = value
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_candidate(self.candidate, self.identity)
            manifest_path.write_text(original, encoding="utf-8")

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
            orchestrator_correlation_id=self.identity.orchestrator_correlation_id,
            github_run_id=self.identity.github_run_id,
            github_run_url=self.identity.github_run_url,
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
            orchestrator_correlation_id=identity.orchestrator_correlation_id,
            github_run_id=identity.github_run_id,
            github_run_url=identity.github_run_url,
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

    def test_published_release_must_be_immutable_and_attested(self) -> None:
        verified = self.verify_published()
        self.assertEqual(verified["immutable"], True)
        self.assertEqual(verified["release_id"], 42)
        self.assertEqual(verified["tag_commit"], self.head)
        self.assertEqual(verified["published_at"], "2026-08-20T22:15:59Z")
        self.assertEqual(
            verified["attested_purl"],
            f"pkg:github/{APPROVED_ASSETS_REPOSITORY}@v0.2.0",
        )
        self.assertEqual(
            verified["attested_assets"], candidate_publication_digests(self.candidate)
        )

    def test_non_immutable_or_malformed_readback_fails_closed(self) -> None:
        release = self.published_release()
        cases: dict[str, dict[str, object]] = {
            "tag-readback-immutable-false": {
                "release_by_tag": {**release, "immutable": False}
            },
            "tag-readback-immutable-missing": {
                "release_by_tag": {
                    key: value for key, value in release.items() if key != "immutable"
                }
            },
            "tag-readback-immutable-string": {
                "release_by_tag": {**release, "immutable": "true"}
            },
            "id-readback-immutable-false": {
                "release_by_id": {**release, "immutable": False}
            },
            "id-readback-immutable-missing": {
                "release_by_id": {
                    key: value for key, value in release.items() if key != "immutable"
                }
            },
            "id-readback-is-a-different-release": {
                "release_by_id": {**release, "id": 43}
            },
            "readbacks-disagree-on-target": {
                "release_by_id": {**release, "target_commitish": "f" * 40}
            },
            "readbacks-disagree-on-published-at": {
                "release_by_id": {
                    **release,
                    "published_at": "2026-08-20T22:16:00Z",
                }
            },
            "classified-release-id-mismatch": {"release_id": 4242},
            "readback-is-a-draft": {"release_by_tag": {**release, "draft": True}},
            "readback-is-unpublished": {
                "release_by_tag": {**release, "published_at": None}
            },
            "readback-is-another-tag": {
                "release_by_tag": {**release, "tag_name": "v0.1.38"}
            },
            "readback-not-an-object": {"release_by_tag": [release]},
            "unapproved-assets-repository": {"assets_repo": "leehack/other-assets"},
            "malformed-tag-commit": {"tag_commit": "not-a-commit"},
        }
        for label, overrides in cases.items():
            with self.subTest(label=label), self.assertRaises(ContractError):
                self.verify_published(**overrides)

    def test_release_attestation_must_cover_the_published_candidate(self) -> None:
        digests = candidate_publication_digests(self.candidate)
        cases: dict[str, dict[str, object]] = {
            "attestation-missing": {"attestation": {}},
            "attestation-null": {"attestation": None},
            "attestation-for-another-tag": {
                "attestation": release_attestation(
                    release_tag="v0.1.38",
                    assets_repo=APPROVED_ASSETS_REPOSITORY,
                    tag_commit=self.head,
                    release_id=42,
                    assets=digests,
                )
            },
            "attestation-for-another-release-id": {
                "attestation": release_attestation(
                    release_tag="v0.2.0",
                    assets_repo=APPROVED_ASSETS_REPOSITORY,
                    tag_commit=self.head,
                    release_id=43,
                    assets=digests,
                )
            },
            "attestation-for-another-commit": {
                "attestation": release_attestation(
                    release_tag="v0.2.0",
                    assets_repo=APPROVED_ASSETS_REPOSITORY,
                    tag_commit="d" * 40,
                    release_id=42,
                    assets=digests,
                )
            },
            "attestation-omits-an-asset": {
                "attestation": release_attestation(
                    release_tag="v0.2.0",
                    assets_repo=APPROVED_ASSETS_REPOSITORY,
                    tag_commit=self.head,
                    release_id=42,
                    assets={
                        name: digest
                        for name, digest in digests.items()
                        if name != "manifest.json"
                    },
                )
            },
            "attestation-digest-mismatch": {
                "attestation": release_attestation(
                    release_tag="v0.2.0",
                    assets_repo=APPROVED_ASSETS_REPOSITORY,
                    tag_commit=self.head,
                    release_id=42,
                    assets={**digests, "manifest.json": "b" * 64},
                )
            },
            "attestation-signed-by-an-impostor": {
                "attestation": self.published_attestation(
                    result_overrides={
                        "signature": {
                            "certificate": {
                                "subjectAlternativeName": "https://evil.example"
                            }
                        }
                    }
                )
            },
        }
        for label, overrides in cases.items():
            with self.subTest(label=label), self.assertRaises(ContractError):
                self.verify_published(**overrides)

    def test_immutable_publication_cli_reports_exact_failures(self) -> None:
        root = Path(self.temporary.name)
        digests = candidate_publication_digests(self.candidate)

        def run(**payloads: object) -> subprocess.CompletedProcess[str]:
            paths: dict[str, Path] = {}
            for key, payload in payloads.items():
                path = root / f"{key}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths[key] = path
            return subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "release_publication_state.py"),
                    "verify-immutable-publication",
                    "--candidate", str(self.candidate),
                    "--assets-repo", APPROVED_ASSETS_REPOSITORY,
                    "--release-tag", "v0.2.0",
                    "--tag-commit", self.head,
                    "--release-id", "42",
                    "--release-json", str(paths["release"]),
                    "--release-by-id-json", str(paths["release_by_id"]),
                    "--attestation-json", str(paths["attestation"]),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        accepted = run(
            release=self.published_release(),
            release_by_id=self.published_release(),
            attestation=self.published_attestation(),
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(json.loads(accepted.stdout)["attested_assets"], digests)

        rejected = run(
            release=self.published_release(immutable=False),
            release_by_id=self.published_release(),
            attestation=self.published_attestation(),
        )
        self.assertEqual(rejected.returncode, 1)
        self.assertIn("is not immutable", rejected.stderr)

        unattested = run(
            release=self.published_release(),
            release_by_id=self.published_release(),
            attestation={},
        )
        self.assertEqual(unattested.returncode, 1)
        self.assertIn("error:", unattested.stderr)


if __name__ == "__main__":
    unittest.main()
