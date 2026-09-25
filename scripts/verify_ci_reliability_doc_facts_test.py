#!/usr/bin/env python3
"""Contract tests for the documentation fact checks in verify_ci_reliability."""

from __future__ import annotations

import unittest

import verify_ci_reliability as verifier

DOC = """# Title

## Publishing

Publication stays blocked until administrators configure
`bridge-assets-publication` with administrator bypass disabled.
Attestation validators come from the trusted workflow commit on `main`; the
requested historical bridge source supplies only exact candidate harness bytes.

```bash
# not a heading
~~~
## still code
```

### Baseline

- `release_tag`: `v0.1.39`
- `release_rebuild`: `0`

## Next

- Publish workflow: `.github/workflows/publish_assets.yml`
  - Requires approval, restricts custom deployment branches to `main`, checks the exact
    `main` branch policy, and must fail closed unless the injected credential is
    non-empty.
- Other workflow: `.github/workflows/ci.yml`
"""

RELATIONS = (
    "validators come from the trusted workflow commit on `main`",
    "historical bridge source supplies only exact candidate harness bytes",
)


def check(text, facts, unit=None, ignore_case=False):
    errors: list[str] = []
    verifier.require_doc_facts(
        "DOC.md", "scope", text, facts, "the facts", errors, unit, ignore_case
    )
    return errors


def publish_item(text):
    return verifier.markdown_list_block(
        "DOC.md", text, "`.github/workflows/publish_assets.yml`", []
    )


class DocFactsTest(unittest.TestCase):
    def test_reflowed_prose_keeps_facts(self) -> None:
        self.assertEqual(check(DOC, RELATIONS), [])
        self.assertEqual(
            check(DOC, ("fail closed unless the injected credential is non-empty",)),
            [],
        )

    def test_missing_fact_is_named(self) -> None:
        errors = check(DOC, ("`bridge-assets-publication`", "`missing-environment`"))
        self.assertEqual(len(errors), 1)
        self.assertIn("DOC.md (scope) must state the facts", errors[0])
        self.assertIn("missing: '`missing-environment`'", errors[0])
        self.assertNotIn("bridge-assets-publication", errors[0])

    def test_case_is_ignored_only_on_request(self) -> None:
        self.assertEqual(len(check(DOC, ("PUBLICATION STAYS",))), 1)
        self.assertEqual(check(DOC, ("PUBLICATION STAYS",), ignore_case=True), [])

    def test_inverted_relation_is_rejected(self) -> None:
        inverted = DOC.replace(
            "validators come from the trusted workflow commit on `main`; the\n"
            "requested historical bridge source supplies only",
            "validators come from the requested historical bridge source; the\n"
            "trusted workflow commit on `main` supplies only",
        )
        self.assertNotEqual(inverted, DOC)
        self.assertEqual(len(check(inverted, RELATIONS)), 1)
        dropped = DOC.replace("supplies only exact", "supplies only")
        self.assertEqual(len(check(dropped, RELATIONS)), 1)

    def test_swapped_value_is_rejected(self) -> None:
        swapped = DOC.replace("branches to `main`", "branches to `release`")
        self.assertNotEqual(swapped, DOC)
        facts = ("restricts custom deployment branches to `main`",)
        self.assertEqual(check(publish_item(DOC), facts, unit="item"), [])
        # `main` still appears in the item, but no longer as the branch value.
        self.assertIn("`main`", publish_item(swapped))
        self.assertEqual(len(check(publish_item(swapped), facts, unit="item")), 1)

    def test_lost_condition_is_rejected(self) -> None:
        inverted = DOC.replace(
            "must fail closed unless the injected credential is",
            "must never fail closed even if the injected credential is",
        )
        self.assertNotEqual(inverted, DOC)
        errors = check(
            publish_item(inverted),
            ("fail closed unless the injected credential is non-empty",),
            unit="item",
        )
        self.assertEqual(len(errors), 1)

    def test_item_unit_binds_key_to_value(self) -> None:
        section = verifier.markdown_section("DOC.md", DOC, "Baseline", [])
        self.assertEqual(check(section, ("`release_tag`: `v0.1.39`",)), [])
        swapped = section.replace("`v0.1.39`", "`0`").replace(
            "`release_rebuild`: `0`", "`release_rebuild`: `v0.1.39`"
        )
        self.assertEqual(len(check(swapped, ("`release_tag`: `v0.1.39`",))), 1)

    def test_section_spans_subsections_and_skips_fenced_headings(self) -> None:
        errors: list[str] = []
        section = verifier.markdown_section("DOC.md", DOC, "Publishing", errors)
        self.assertEqual(errors, [])
        self.assertIn("`release_tag`", section)
        self.assertIn("# not a heading", section)
        # A tilde line does not close a backtick fence.
        self.assertIn("## still code", section)
        self.assertNotIn("Publish workflow", section)
        self.assertEqual(verifier.markdown_section("DOC.md", DOC, "Absent", errors), "")
        self.assertEqual(errors, ["DOC.md is missing the Markdown section: Absent"])

    def test_duplicate_section_title_is_rejected(self) -> None:
        errors: list[str] = []
        doubled = DOC + "\n## Publishing\n\nShadow.\n"
        section = verifier.markdown_section("DOC.md", doubled, "Publishing", errors)
        self.assertIn("with administrator bypass disabled", section)
        self.assertEqual(len(errors), 1)
        self.assertIn("2 Markdown sections titled Publishing", errors[0])

    def test_list_block_keeps_nested_items_only(self) -> None:
        errors: list[str] = []
        block = publish_item(DOC)
        self.assertIn("restricts custom deployment", block)
        self.assertNotIn("ci.yml", block)
        verifier.markdown_list_block("DOC.md", DOC, "restricts custom", errors)
        self.assertEqual(
            errors, ["DOC.md is missing the list item naming restricts custom"]
        )

    def test_paragraph_with_list_counts_top_level_items(self) -> None:
        text = (
            "Required credentials:\n\n- `ONE` (nested\n  detail)\n  - sub\n\n"
            "Every request follows.\n"
        )
        errors: list[str] = []
        block = verifier.markdown_paragraph_with_list(
            "DOC.md", text, "credentials", errors
        )
        self.assertEqual(errors, [])
        self.assertNotIn("Every request", block)
        self.assertEqual(verifier.markdown_top_level_item_count(block), 1)
        self.assertEqual(
            verifier.markdown_top_level_item_count(block + "\n- `TWO`\n"), 2
        )

    def test_extra_credential_paragraph_is_detected(self) -> None:
        text = (
            "- `WEBGPU_BRIDGE_ASSETS_PAT` (write access).\n\n"
            "`LLAMADART_RELEASE_TOKEN` must also be stored in the environment.\n\n"
            "Every request supplies a `candidate_run_id` and `MAX_TOKENS_LIMIT`.\n"
        )
        found = set(verifier.SECRET_LIKE_IDENTIFIER.findall(text))
        self.assertEqual(
            found - {verifier.PUBLICATION_PAT_NAME}, {"LLAMADART_RELEASE_TOKEN"}
        )


if __name__ == "__main__":
    unittest.main()
